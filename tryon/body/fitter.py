"""Fit a SMPL-X body to MediaPipe landmarks, then track it.

This is the 3D prior the whole rebuild rests on. The previous system placed
garments from four 2D anchors, which cannot express a limb rotation:
measured on real frames, an outstretched arm needs the upper arm turned by
-72 degrees and the forearm by -47 while the torso stays at 0, and one
affine transform carries one rotation. A body made of joints does not have
that problem.

The method is SMPLify (Bogo et al. 2016; Kolotouros et al. arXiv:1909.12828):
minimise the distance between projected model joints and detected 2D
keypoints, with a prior keeping the pose plausible.

Two measurements shaped the design, both taken before any of this was
written:

* SMPL-X runs at 6.04 ms on CPU against 38.8 ms on MPS - MPS falls back to
  a view-op implementation for padding inside the skinning, so CPU wins.
  This module therefore stays on CPU.
* A cold fit (30 iterations) costs 418 ms, i.e. 2.4 fps - far too slow per
  frame. Warm-started from the previous frame, 3 iterations cost 42 ms
  (23.9 fps) and the fit holds as the body moves. So the class is stateful:
  `fit` once when a person appears, `track` every frame after.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

# Straight from MediaPipe, not from `tryon.views`: the 3D path must not
# depend on the 2D warping stack it replaces.
import cv2
import numpy as np
import torch
from tryon import paths
from tryon.body.landmarks import (
    _IDS,
    _TORSO,
    JOINT_MAP,
    VISIBILITY_FLOOR,
    _all_depth_bones,
    _scaled_points,
    landmark_yaw,
    landmarks_to_depths,
    landmarks_to_targets,
    yaw_difference,
)

SMPLX_ROOT = paths.SMPLX_ROOT

# Weight of the bone-depth term relative to the 2D term. The 2D residual
# is in px^2; the depth residual (metres) is scaled by the camera's
# px/m so a 10 cm depth error costs as much as a 10 cm in-plane one
# times this factor.
DEPTH_WEIGHT = 0.5
# Camera scale (px per metre) the fit runs at: landmarks are rescaled so
# the person's torso implies this scale, fitted, and the camera scaled
# back. None fits in the caller's pixels. See `BodyFitter.fit`.
WORK_SCALE = 120.0
# Use MediaPipe's world-landmark torso yaw (`landmark_yaw`) as a facing
# hypothesis and tie-breaker, and a refit trigger in live.track_step.
USE_LANDMARK_YAW = True
# Silhouette term in `track` (see BodyFitter.set_silhouette): weight, and
# how many of the last iterations of a frame carry it. 0 = off. Against the
# authors' registration (tools/evaluate/vs_registration.py, previous
# frame's parse as live has it): weights 0 / 0.3 / 1 / 3 / 10 / 30 on
# 00190_upper IoU 0.652 / 0.658 / 0.666 / 0.678 / 0.663 / 0.617; at 3 on
# the three clips not used to pick it 0.652 -> 0.662, 0.735 -> 0.723,
# 0.591 -> 0.603, body+garment vs person 0.706 -> 0.725, 0.767 -> 0.780,
# 0.733 -> 0.756. ~25 ms a frame.
SILHOUETTE_WEIGHT = 3.0
SILHOUETTE_ITERATIONS = 3
# Skin term (BodyFitter.set_skin); 0 = off.
SKIN_WEIGHT = 0.0
# Candidates within this factor of the best loss are "equally good" and
# the one facing MediaPipe's way is taken.
YAW_LOSS_SLACK = 1.3
# Keypoint refinement steps after an image regressor's seed
# (`BodyFitter.fit_from_regression`).
REGRESSION_ITERATIONS = 50
# Where a regressed body is placed in the image: "regressor" (its own
# projected joints) or "keypoints" (camera fitted to MediaPipe).
REGRESSION_CAMERA = "regressor"
# `track` shifts the camera by the hips' mean residual beyond a dead zone
# before iterating (see `BodyFitter._recentre`). 15 release clips
# (tools/evaluate/tracking_jitter.py), dead zone 0.02 / 0.04 / 0.08 m:
# reprojection 2.78 -> 2.23 / 2.33 / 2.62% of height, on-screen
# acceleration 24.55 -> 24.39 / 24.46 / 24.37; all keep the synthetic
# turn tests (no dead zone, or all four torso joints, mirrored the turn).
RECENTRE_TRACK = True
# metres of hip offset left to the optimiser (see `_recentre`)
RECENTRE_DEAD_ZONE = 0.02


_SMPLX_MODELS = {}  # (models root, gender) -> the loaded model, copied per BodyFitter


class BodyFitter:
    """A SMPL-X body that is fitted once and then tracked."""

    # Weight of the shape prior in the 2D fit. Keypoints constrain bone
    # lengths, never girth, so with a weak prior the shape drifts wide:
    # on camera 0004 (a wearer whose scan body is 24 cm across the chest,
    # 1.76 m tall) the 2D fit at 20 gave 40 cm and 1.85 m, and a garment
    # bound to the scan body ended up inside the live one. Swept on that
    # clip (reprojection / T-pose chest width / height / posed garment
    # inside the body): 20 -> 7.7 px / 40 cm / 1.85 m / 50%; 100 -> 8.8 /
    # 36 / 1.83 / 24%; 400 -> 10.0 / 30 / 1.78 / 13%; 1600 -> 11.1 / 23 /
    # 1.75 / 9%. The wearer's true body (the scan) is 24 cm / 1.76 m, so
    # 1600 reproduces it - but on the profile cameras of the annotated
    # takes it pushed reprojection to 26-30 px (tests/test_body_fit_real_
    # photos, threshold 25), so 400 is the compromise: near-average girth
    # (chest 30 cm, betas within 0.7) at 10 px on camera 0004.
    SHAPE_PRIOR = 400.0
    # Arm-only refinement of a regressed pose toward MediaPipe's 2D elbows and wrists
    # (`_refine_arms`, tools/evaluate/arm_pose.py); off until it is measured.
    ARM_REFINE = False
    ARM_GATE = 0.12  # torso lengths of elbow / wrist disagreement before an arm is touched
    ARM_VISIBILITY = 0.5  # MediaPipe visibility of a landmark worth following
    ARM_BLEND = 1.0  # how far toward MediaPipe's point the target sits (1: on it)
    ARM_PRIORS = (0.3, 1.0, 0.1)  # per rad: shoulder, elbow non-hinge, elbow flexion
    POSE_PRIOR = 60.0
    # Centre of the pose prior (1, 63): None = rest; `fit_from_regression`
    # sets it to the image regressor's pose, so the keypoint term refines
    # that pose instead of pulling the body back toward a T-pose.
    pose_anchor = None
    # Centre of the shape prior (10 betas) - the fit collapses onto it at
    # WORK_SCALE, so it IS the body shape a monocular fit ends up with.
    # None = SMPL-X's own mean (zeros). A population mean (leave-one-
    # subject-out over the scan subjects) left the registration comparison
    # unchanged (garment IoU 0.650 -> 0.649) and was removed.
    shape_mean = None

    def _shape_mean(self):
        if self.shape_mean is None:
            return torch.zeros_like(self.betas)
        return torch.as_tensor(np.asarray(self.shape_mean, np.float32)).reshape(self.betas.shape)

    def __init__(self, gender="neutral", models_root=SMPLX_ROOT, threads=8, shape_frames=30):
        import smplx

        torch.set_num_threads(threads)
        # The model file is read and decompressed once per process (~0.5-1.5
        # s a time; garment binding built three fitters), every fitter gets
        # its own deep copy of it.
        key = (str(models_root), gender)
        if key not in _SMPLX_MODELS:
            model = smplx.create(
                str(models_root),
                model_type="smplx",
                gender=gender,
                use_pca=False,
                batch_size=1,
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            _SMPLX_MODELS[key] = model
        self.model = copy.deepcopy(_SMPLX_MODELS[key])
        self.body_pose = torch.zeros(1, 63, requires_grad=True)
        self.global_orient = torch.zeros(1, 3, requires_grad=True)
        self.betas = torch.zeros(1, 10, requires_grad=True)
        # Weak perspective: a scale and a 2D offset, which is what SMPLify
        # uses when the camera is unknown. Solving a full intrinsic matrix
        # from one view is under-determined and not needed to place a
        # garment on the person who is already in the frame.
        self.scale = torch.tensor([250.0], requires_grad=True)
        self.translation = torch.zeros(1, 2, requires_grad=True)
        self._joint_order = sorted(JOINT_MAP)
        self._smplx_indices = [JOINT_MAP[index] for index in self._joint_order]
        # Bone-depth targets for the current frame (see landmarks_to_depths),
        # set by fit()/track(); None = no depth term.
        self._depth_pairs = [(self._joint_order.index(a), self._joint_order.index(b)) for a, b in _all_depth_bones()]
        self._depth_targets = None
        self._depth_weights = None
        # The 20-wide blend basis the library actually uses. `model.shapedirs`
        # is the 10-wide shape-only slice; the full one lives beside it.
        self._shapedirs_full = torch.cat([self.model.shapedirs, self.model.expr_dirs], dim=-1)
        self._optimiser = None
        self._optimiser_key = None
        # Shape is estimated over the first `shape_frames` tracked frames
        # and then held. Twelve skeleton keypoints carry almost no shape
        # information - measured: the same person fitted from four cameras
        # got four different shapes (beta spread 0.75), and holding one
        # shape across all four cost 1.0 px. Per-frame shape is noise
        # chasing pixels. Accumulating it is VIBE's sequence pooling.
        self.shape_frames = int(shape_frames)
        self._shape_history = []
        self.shape_locked = False

    # A 3D world offset (1, 3) applied to vertices and joints, set by
    # `fit_mesh` (and by GarmentRenderer for the scan-frame body); None for
    # the live camera fit, whose translation is the 2D weak-perspective one.
    mesh_offset = None

    def _forward(self):
        """The library's full forward: the skinned mesh. Used for vertices.

        Without autograd the result is cached until a parameter changes
        (the tensors' in-place version counters): a live frame asked for
        the same posed body up to ~8 times, ~6-12 ms each. Callers get
        fresh tensors, so the cache cannot be edited through them."""
        if torch.is_grad_enabled():
            return self._forward_uncached()
        # the tensors themselves are kept (not their ids), so a replaced
        # parameter can never alias a freed one
        inputs = (self.betas, self.body_pose, self.global_orient, self.mesh_offset)
        versions = tuple(None if t is None else t._version for t in inputs)
        cached = getattr(self, "_forward_cache", None)
        if cached is None or versions != cached[1] or any(a is not b for a, b in zip(inputs, cached[0])):
            output = self._forward_uncached()
            cached = (inputs, versions, output.vertices.detach(), output.joints.detach())
            self._forward_cache = cached
        return SimpleNamespace(vertices=cached[2].clone(), joints=cached[3].clone())

    def _forward_uncached(self):
        output = self.model(
            betas=self.betas,
            body_pose=self.body_pose,
            global_orient=self.global_orient,
        )
        if self.mesh_offset is not None:
            output.vertices = output.vertices + self.mesh_offset
            output.joints = output.joints + self.mesh_offset
        return output

    def fit_mesh(self, target_vertices, iterations=500, learning_rate=0.05):
        """Fit shape, pose and orientation to a SMPL-X-topology mesh
        (10,475 vertices) given in 3D - the scan-frame body the
        Gaussian-Garments trajectory files carry (`obstacle[0]`, the
        SMPL-X fit of the frame the garment template was scanned in).
        Vertex-to-vertex, so it is well posed: the target came out of the
        same model. Returns the mean vertex residual in metres. The 3D
        offset between model and target is solved as `self.mesh_offset`
        (the fitter's own translation is a 2D image one)."""
        target = torch.as_tensor(np.asarray(target_vertices), dtype=torch.float32)
        if target.shape != (self.model.get_num_verts(), 3):
            raise ValueError(f"expected ({self.model.get_num_verts()}, 3) vertices, got {tuple(target.shape)}")
        with torch.no_grad():
            template = self.model(betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 63), global_orient=torch.zeros(1, 3)).vertices[0]
        self.mesh_offset = (target.mean(dim=0) - template.mean(dim=0))[None].clone().requires_grad_(True)
        residual = float("nan")

        def run(params, steps, lr):
            nonlocal residual
            optimiser = torch.optim.Adam(params, lr)
            for step in range(int(steps)):
                if step == int(steps * 0.6):
                    for group in optimiser.param_groups:
                        group["lr"] = lr * 0.2
                optimiser.zero_grad()
                vertices = self._forward().vertices[0]  # includes mesh_offset
                data = ((vertices - target) ** 2).sum(dim=1).mean()
                # priors only break ambiguity: the data term is in m^2 (1e-4 at 1 cm)
                loss = data + 1e-6 * (self.body_pose**2).mean() + 1e-6 * (self.betas**2).mean()
                loss.backward()
                optimiser.step()
                residual = float(data.detach().sqrt())

        # Rigid first (orientation + offset), then everything: solving all
        # at once from the rest pose left a synthetic body 2 cm off after
        # 400 steps, the joints chasing an orientation that had not settled.
        run([self.global_orient, self.mesh_offset], max(int(iterations) // 4, 50), learning_rate)
        run([self.betas, self.body_pose, self.global_orient, self.mesh_offset], iterations, learning_rate)
        self._optimiser = None
        return residual

    def joints(self, indices=None):
        """The fitted joints without skinning the mesh - the tracking path.

        `lbs` skins 10,475 vertices to return 55 joints, but computes those
        joints BEFORE skinning: regress rest joints from the shaped template,
        then rigidly transform them down the kinematic chain. Doing only that
        is 2.2 ms per forward+backward against 9.7 - the difference between a
        choppy live view and a smooth one - and it is the library's own
        formula, so it matches to 1e-7 m.

        Matching depends on two inputs that the model's public attributes do
        NOT expose the way they are used, both found by capturing what the
        library actually passes to `lbs`:

        * shape is blended through a 20-wide `shapedirs` - betas plus
          expression - not the 10-wide tensor `model.shapedirs` returns.
        * the hands are posed at `left_hand_mean` / `right_hand_mean`, not
          at zero. Zero is not the rest hand. Leaving the mean out bent the
          wrists 4 cm - exactly the joints a sleeve hangs from.
        """
        posed, _ = self._chain()
        if self.mesh_offset is not None:
            posed = posed + self.mesh_offset
        return posed[0, self._smplx_indices if indices is None else indices]

    def _chain(self):
        """(posed joints, joint transforms) before the world offset: rest joints
        from the shaped template, then the kinematic chain.

        Without autograd the result is cached until a parameter changes (the
        tensors' in-place version counters, like `_forward`): a live frame
        asked for it ~5 times and each is 6 `batch_rigid_transform` calls.
        The cached tensors are never handed out: callers add the offset into
        new tensors or clone."""
        inputs = (self.betas, self.body_pose, self.global_orient)
        if torch.is_grad_enabled():
            return self._chain_uncached(*inputs)
        versions = tuple(t._version for t in inputs)
        cached = getattr(self, "_chain_cache", None)
        if cached is not None and versions == cached[1] and all(a is b for a, b in zip(inputs, cached[0])):
            return cached[2]
        result = self._chain_uncached(*inputs)
        self._chain_cache = (inputs, versions, result)
        return result

    def _chain_uncached(self, betas, body_pose, global_orient):
        from smplx.lbs import (
            batch_rigid_transform,
            batch_rodrigues,
            blend_shapes,
            vertices2joints,
        )

        model = self.model
        shape = torch.cat([betas, model.expression], dim=-1)
        shaped = model.v_template + blend_shapes(shape, self._shapedirs_full)
        rest_joints = vertices2joints(model.J_regressor, shaped)
        pose = torch.cat(
            [
                global_orient,
                body_pose,
                model.jaw_pose,
                model.leye_pose,
                model.reye_pose,
                model.left_hand_pose + model.left_hand_mean,
                model.right_hand_pose + model.right_hand_mean,
            ],
            dim=1,
        )
        rotations = batch_rodrigues(pose.view(-1, 3)).view(1, -1, 3, 3)
        posed, transforms = batch_rigid_transform(rotations, rest_joints, model.parents)
        return posed, transforms

    def joint_transforms(self):
        """The 55 per-joint 4x4 transforms of the current pose, relative to rest.

        These are what skinning applies: a vertex with skinning weights `w`
        moves by `sum_j w_j A_j`. The body model uses them on its own 10,475
        vertices; `bind.SkinnedGarment` uses the same matrices on a garment.
        Same chain as `joints()`, kept in one place so the two cannot drift.
        """
        _, transforms = self._chain()
        transforms = transforms[0].clone()  # never a view of the cache
        if self.mesh_offset is not None:
            # `_forward` adds the world offset AFTER skinning; the skinning
            # weights sum to one, so adding it to every joint's translation
            # is the same thing - and a garment skinned by these transforms
            # lands in the same frame as `vertices()` (measured before this:
            # a fitter set from `fit_mesh` put the garment 1.65 m below its
            # own body, so the cloth solver collided against nothing).
            transforms[:, :3, 3] = transforms[:, :3, 3] + self.mesh_offset
        return transforms

    def _project(self, joints):
        """Weak-perspective projection: image y grows downward, model y up."""
        flipped = joints[..., :2] * torch.tensor([1.0, -1.0])
        return flipped * self.scale + self.translation

    def _parameters(self, shape=True):
        """What the optimiser may move. Shape only during the cold fit.

        VIBE predicts shape per frame and then average-pools it over the
        whole sequence, because a body does not change shape between frames.
        Here that is enforced rather than averaged: `track` leaves beta out
        of the optimiser entirely, so it cannot drift and cannot be used to
        explain a pose change as a shape change and back again.
        """
        parameters = [self.body_pose, self.global_orient, self.scale, self.translation]
        if shape:
            parameters.insert(2, self.betas)
        return parameters

    def _step(self, targets, weights, learning_rate, shape=True):
        wanted = (learning_rate, shape)
        if self._optimiser is None or self._optimiser_key != wanted:
            self._optimiser = torch.optim.Adam(self._parameters(shape), learning_rate)
            self._optimiser_key = wanted
        self._optimiser.zero_grad()
        joints = self.joints()
        projected = self._project(joints)
        residual = (projected - targets) ** 2
        data = (residual.sum(dim=1) * weights).sum() / weights.sum().clamp(min=1e-6)
        if self._depth_targets is not None and float(self._depth_weights.sum()) > 0:
            # Bone depth from MediaPipe's 3D estimate (DEPTH_BONES), in
            # pixel-equivalent units so it trades off against the 2D term.
            a = torch.tensor([p[0] for p in self._depth_pairs])
            b = torch.tensor([p[1] for p in self._depth_pairs])
            dz = joints[b, 2] - joints[a, 2]
            depth = ((dz - self._depth_targets) ** 2 * self._depth_weights).sum() / self._depth_weights.sum()
            data = data + DEPTH_WEIGHT * self.scale.detach() ** 2 * depth
        # Keep the pose near rest and the shape near average. Without this a
        # 63-dimensional pose will contort to hit 13 noisy keypoints.
        #
        # The global orientation needs its own term, and it is the one that
        # matters most. Under weak perspective a body tipped toward the
        # camera projects almost exactly like an upright one - the classic
        # monocular depth ambiguity (Bogo et al. 2016) - so the 2D loss is
        # indifferent between them. Measured without this term against an
        # upright synthetic body with exact ground truth: reprojection 12.9 px,
        # yet the fit leaned 65 degrees and every joint was 378 mm off. A
        # standing person is upright; claiming otherwise has to cost.
        #
        # Pitch and roll only. Yaw is free: a person can face any way, and
        # anchoring it at zero stalled the annotated back camera at 38 px,
        # halfway round. Pitch and roll are the axes a standing person does
        # not rotate about, and they are the depth ambiguity's escape route.
        upright = self.global_orient[:, [0, 2]]
        anchor = self.pose_anchor if self.pose_anchor is not None else 0.0
        prior = (
            self.POSE_PRIOR * ((self.body_pose - anchor) ** 2).mean()
            + self.SHAPE_PRIOR * ((self.betas - self._shape_mean()) ** 2).mean()
            + 400.0 * (upright**2).mean()
        )
        if self._silhouette_active and SILHOUETTE_WEIGHT > 0 and self._silhouette is not None:
            prior = prior + SILHOUETTE_WEIGHT * self._silhouette_loss()
        if self._silhouette_active and SKIN_WEIGHT > 0 and self._skin_points is not None:
            prior = prior + SKIN_WEIGHT * self._skin_loss()
        loss = data + prior
        loss.backward()
        self._optimiser.step()
        return float(data.detach())

    def _initial_scale(self, target_tensor, torso_weights, fallback=250.0, rest_joints=None):
        """Camera scale seed: observed torso height over the rest torso's.

        A fixed seed of 250 cannot be moved far by the camera stage (Adam,
        step 2.0, 100 steps), so a person 1,430 px tall in a 1,698 px frame
        - true scale ~800 - was fitted at 412 with 102 px reprojection
        (reference-frame ablation, 00152_outer frame 0). The vertical
        shoulder-to-hip extent is used because it does not shrink when the
        wearer turns to profile, unlike shoulder width. Called with the
        pose at rest and the yaw hypothesis already set. `rest_joints`
        (precomputed joints of that rest pose) skips the kinematic chain.
        """
        visible = torso_weights > 0
        if int(visible.sum()) < 2:
            return fallback
        with torch.no_grad():
            saved = (self.scale.detach().clone(), self.translation.detach().clone())
            self.scale.fill_(1.0)
            self.translation.zero_()
            rest = self._project(self.joints() if rest_joints is None else rest_joints)
            self.scale.copy_(saved[0])
            self.translation.copy_(saved[1])
        rest_height = float(rest[visible, 1].max() - rest[visible, 1].min())
        target_height = float(target_tensor[visible, 1].max() - target_tensor[visible, 1].min())
        if rest_height <= 1e-6 or target_height <= 1.0:
            return fallback
        return target_height / rest_height

    def _fit_camera(self, target_tensor, weight_tensor, torso_weights, camera_iterations):
        """Stage 1: scale and translation from the torso alone, pose fixed.

        The torso is rigid, so with the CORRECT facing this is well-
        determined by four points. With the WRONG facing it is not: a torso
        turned 180 degrees from the rest pose used here is not a scaled,
        translated copy of a forward-facing rest torso, so no scale and
        translation reproduce it and the loss floors well above zero however
        long this runs (measured: 10,953 px^2 after 2,000 iterations on the
        annotated camera 0052, where the real yaw is 178 degrees - one
        pixel-perfect converged run and one that isn't, and only the global
        orientation set before this stage tells them apart). This is called
        once per yaw hypothesis in `fit`, not once overall, so each stage 2
        candidate starts from a camera that is actually consistent with its
        own facing.
        """
        camera = torch.optim.Adam([self.scale, self.translation], 2.0)
        for _ in range(max(int(camera_iterations), 1)):
            camera.zero_grad()
            residual = ((self._project(self.joints()) - target_tensor) ** 2).sum(dim=1)
            (residual * torso_weights).sum().div(torso_weights.sum().clamp(min=1e-6)).backward()
            camera.step()

    def fit(self, points, iterations=200, learning_rate=0.02, camera_iterations=100, orientation_hypotheses=None, prefer_yaw="landmarks"):
        """Cold fit in a normalised frame - see `_fit_working` for the method.

        The landmarks are rescaled so the person's camera scale is
        WORK_SCALE px/m, fitted there, and the camera scaled back. Fitting
        the same MediaPipe landmarks of the reference frame (00152_outer,
        1,698 px panel) at different working sizes, reprojection at panel
        scale / body silhouette IoU: 1,698 px 78.0 / 0.57, 870 px 48.5 /
        0.56, 480 px 32.8 / 0.65, 240 px 18.1 / 0.74, 120 px 19.9 / 0.74
        (the authors' own body scores 0.72). MediaPipe's input size made
        no difference (480 vs 870 px within 2 px), so it is the fit, and
        the loss/prior balance alone does not explain it (normalising the
        data term by scale^2 made 870 px worse). The returned loss is px^2
        of the caller's image.
        """
        if isinstance(prefer_yaw, str):
            prefer_yaw = landmark_yaw(points) if USE_LANDMARK_YAW else None
        factor = self._working_factor(points)
        self._work_factor = factor
        loss = self._fit_working(_scaled_points(points, factor), iterations, learning_rate, camera_iterations, orientation_hypotheses, prefer_yaw)
        self._rescale_camera(1.0 / factor)
        return loss / factor**2

    def fit_from_regression(self, points, prediction, iterations=REGRESSION_ITERATIONS, learning_rate=0.02, camera_iterations=100, camera=None):
        """Cold fit seeded by an image regressor (`tryon.body.regressor`).

        Pose, shape and facing come from the regressor, which sees the
        whole image; only the camera (torso, stage 1 of `_fit_working`)
        and a short keypoint refinement come from MediaPipe, with the pose
        prior centred on the regressed pose (`pose_anchor`) so the
        refinement aligns rather than replaces it. `prediction`: a dict
        with global_orient (3,), body_pose (63,), betas (10,) and, for
        `camera` "regressor", joints_2d (22, 2). `camera`: None =
        REGRESSION_CAMERA.
        """
        camera = REGRESSION_CAMERA if camera is None else camera
        landmarks = points
        if points is None:
            # MediaPipe lost the wearer (e.g. an arm or object across the
            # torso); the regressor still saw them. No landmark is used.
            points = np.zeros((33, 5), np.float32)
            camera = "regressor"
        factor = self._working_factor(points)
        self._work_factor = factor
        points = _scaled_points(points, factor)
        targets, weights = landmarks_to_targets(points)
        self._set_depths(points)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        torso_weights = weight_tensor * torch.tensor([1.0 if index in _TORSO else 0.0 for index in self._joint_order])
        visible = weights > 0
        pose = torch.tensor(np.asarray(prediction["body_pose"], np.float32)).reshape(self.body_pose.shape)
        with torch.no_grad():
            self.body_pose.copy_(pose)
            self.global_orient.copy_(torch.tensor(np.asarray(prediction["global_orient"], np.float32)).reshape(self.global_orient.shape))
            # each prediction's own betas: holding the first 30 frames' mean
            # (RTV, Wu et al. 2025) scored 0.727 against 0.730 on the release
            # videos, where the wearer wears the garment being scored
            betas = np.asarray(prediction["betas"], np.float32)[:self.betas.shape[1]]
            self.betas.copy_(torch.tensor(betas).reshape(self.betas.shape))
            self.scale.copy_(torch.tensor([self._initial_scale(target_tensor, torso_weights)]))
            if visible.any():
                self.translation.copy_(torch.tensor(targets[visible].mean(axis=0), dtype=torch.float32))
        self.pose_anchor = pose.clone()
        self.shape_mean = self.betas.detach().numpy().ravel().copy()
        self._shape_history = []
        self.shape_locked = False
        self._optimiser = None
        if camera == "regressor" and prediction.get("joints_2d") is not None:
            # The regressor's own image placement: least-squares scale and
            # translation of our 22 body joints onto its projected ones
            # (its camera is perspective, ours weak), no MediaPipe at all.
            with torch.no_grad():
                ours = self._project(self.joints(list(range(22)))).numpy()
                ours = (ours - self.translation.detach().numpy()) / float(self.scale)
            wanted = np.asarray(prediction["joints_2d"], np.float64)
            # Only the joints it places inside the image: for a wearer cut
            # off at the waist the legs are its guess, and fitting them
            # too put the garment 20-30 px high on the torso it can see.
            inside = np.ones(len(wanted), bool)
            if prediction.get("image_size") is not None:
                h, w = prediction["image_size"]
                inside = (wanted[:, 0] >= 0) & (wanted[:, 0] < w) & (wanted[:, 1] >= 0) & (wanted[:, 1] < h)
                if inside.sum() < 4:
                    inside[:] = True
            ours_in, wanted_in = ours[inside], wanted[inside]
            a, b = ours_in - ours_in.mean(axis=0), wanted_in - wanted_in.mean(axis=0)
            s = float((a * b).sum() / max((a * a).sum(), 1e-9))
            with torch.no_grad():
                self.scale.copy_(torch.tensor([s], dtype=torch.float32))
                self.translation.copy_(torch.tensor(wanted_in.mean(axis=0) - s * ours_in.mean(axis=0), dtype=torch.float32)[None])
            if self.ARM_REFINE and landmarks is not None:
                self._refine_arms(np.asarray(landmarks, np.float64))
            return float(((s * ours_in + self.translation.detach().numpy() - wanted_in) ** 2).sum(axis=1).mean())
        self._fit_camera(target_tensor, weight_tensor, torso_weights, camera_iterations)
        # Root and camera only: releasing the pose (or shape) to the
        # keypoints undid the regressor - annotated photos, 8 views,
        # PA-MPJPE 50 mm seeded, 83 mm after 10 all-parameter steps, 69 mm
        # at a tenth of the step; root + camera keeps 50 mm (reprojection
        # 14.5 -> 13.8 px; our keypoint fit reaches 9 px but 94 mm).
        loss = 0.0
        camera = torch.optim.Adam([self.global_orient, self.scale, self.translation], learning_rate)
        for _ in range(int(iterations)):
            camera.zero_grad()
            residual = ((self._project(self.joints()) - target_tensor) ** 2).sum(dim=1)
            data = (residual * weight_tensor).sum() / weight_tensor.sum().clamp(min=1e-6)
            data.backward()
            camera.step()
            loss = float(data.detach())
        self._optimiser = None
        self._rescale_camera(1.0 / factor)
        return loss / factor**2

    def _refine_arms(self, landmarks):
        """Move each arm's shoulder and elbow rotations toward MediaPipe's visible elbow and wrist
        where the fitted ones disagree by more than `ARM_GATE` torso lengths (camera already placed).
        Returns the arms touched."""
        from tryon.body.regressor import ARM_JOINTS, refine_arm

        if min(landmarks[i, 2] for i in (11, 12, 23, 24)) < VISIBILITY_FLOOR:
            return []
        length = 0.5 * float(np.linalg.norm(landmarks[11, :2] - landmarks[23, :2]) + np.linalg.norm(landmarks[12, :2] - landmarks[24, :2]))
        if length < 1.0:
            return []
        with torch.no_grad():
            posed, transforms = self._chain()
            offset = torch.zeros(3) if self.mesh_offset is None else self.mesh_offset[0]
            posed = (posed[0, :22] + offset).double().numpy()
            rotations = transforms[0, :22, :3, :3].double().numpy()
            camera = (float(self.scale), self.translation.detach().double().numpy().reshape(2))
        shoulder_prior, twist_prior, flexion_prior = self.ARM_PRIORS
        flipped = False
        touched = []
        if min(landmarks[15, 2], landmarks[16, 2]) >= self.ARM_VISIBILITY:
            # MediaPipe's left / right wrists nearer the other way round than NLF's: whose labels hold is unknown
            keep = np.linalg.norm(self._pixels(posed[20], camera) - landmarks[15, :2]) + np.linalg.norm(self._pixels(posed[21], camera) - landmarks[16, :2])
            swap = np.linalg.norm(self._pixels(posed[21], camera) - landmarks[15, :2]) + np.linalg.norm(self._pixels(posed[20], camera) - landmarks[16, :2])
            flipped = swap < 0.5 * keep
        if flipped:
            return []
        for name, (collar, shoulder, elbow, wrist, sign), (mp_elbow, mp_wrist) in zip(ARM_JOINTS, ARM_JOINTS.values(), ((13, 15), (14, 16))):
            targets = {key: landmarks[m, :2] if landmarks[m, 2] >= self.ARM_VISIBILITY else None for key, m in (("elbow", mp_elbow), ("wrist", mp_wrist))}
            seen = {"elbow": elbow, "wrist": wrist}
            gap = [np.linalg.norm(self._pixels(posed[seen[k]], camera) - xy) / length for k, xy in targets.items() if xy is not None]
            if not gap or max(gap) < self.ARM_GATE:
                continue
            if self.ARM_BLEND < 1.0:
                targets = {k: None if xy is None else (lambda here: here + self.ARM_BLEND * (xy - here))(self._pixels(posed[seen[k]], camera)) for k, xy in targets.items()}
            a_sh = self.body_pose[0, (shoulder - 1) * 3: shoulder * 3].detach().double().numpy()
            a_el = self.body_pose[0, (elbow - 1) * 3: elbow * 3].detach().double().numpy()
            frames = {"R_collar": rotations[collar], "shoulder": posed[shoulder], "a_shoulder": a_sh, "a_elbow": a_el,
                      "o_elbow": rotations[shoulder].T @ (posed[elbow] - posed[shoulder]),
                      "o_wrist": rotations[elbow].T @ (posed[wrist] - posed[elbow])}
            new_sh, new_el, before, after = refine_arm(frames, targets, length, camera, sign, shoulder_prior, twist_prior, flexion_prior)
            if after < before:
                with torch.no_grad():
                    self.body_pose[0, (shoulder - 1) * 3: shoulder * 3] = torch.tensor(new_sh, dtype=torch.float32)
                    self.body_pose[0, (elbow - 1) * 3: elbow * 3] = torch.tensor(new_el, dtype=torch.float32)
                touched.append(name)
        return touched

    @staticmethod
    def _pixels(joint, camera):
        return joint[:2] * (1.0, -1.0) * camera[0] + camera[1]

    def refit(self, points, extra_hypotheses=(), prefer_yaw=None):
        """Cold fit that may only improve on the current pose.

        For a tracked body whose reprojection suddenly jumps (a fast turn,
        a camera cut): warm-started tracking turns only gradually and sat
        80-100 degrees short of a back view for ten frames. The current
        yaw is one of the hypotheses, and the old state is restored unless
        the refit reprojects better. Returns True when adopted. Measured on
        the four-camera turn clip (/tmp/e2e_turn.mp4, cuts every 10 frames):
        yaw after each cut 39 -> 146 -> 172 degrees on the back camera
        (true ~178) instead of 71 -> 100, reprojection 27 -> 5 px.
        """
        before = self.reprojection_error(points)
        parameters = (self.body_pose, self.global_orient, self.betas, self.scale, self.translation)
        saved = [parameter.detach().clone() for parameter in parameters]
        state = (getattr(self, "_work_factor", 1.0), list(self._shape_history), self.shape_locked)
        yaw = float(self.global_orient[0, 1])
        hypotheses = (yaw, 0.0, float(np.pi), float(np.pi / 2), float(-np.pi / 2)) + tuple(extra_hypotheses)
        old_yaw = yaw
        self.fit(points, orientation_hypotheses=hypotheses)
        after = self.reprojection_error(points)
        if after < before:
            return True
        if prefer_yaw is not None:
            # Called because the facing disagrees with MediaPipe's: adopt a
            # refit that agrees better if it reprojects about as well.
            new_yaw = float(self.global_orient.detach()[0, 1])
            closer = yaw_difference(new_yaw, prefer_yaw) + np.radians(10) < yaw_difference(old_yaw, prefer_yaw)
            if closer and after <= before * YAW_LOSS_SLACK + 2.0:
                return True
        with torch.no_grad():
            for parameter, value in zip(parameters, saved):
                parameter.copy_(value)
        self._work_factor, self._shape_history, self.shape_locked = state[0], state[1], state[2]
        self._optimiser = None
        return False

    def _zero_pose_joints(self):
        """`joints()` at zero pose and zero betas, computed once: it depends
        only on the model and the (normally absent) mesh offset, never on the
        fitted parameters, and the fitted parameters are not touched."""
        offset = self.mesh_offset
        key = (offset, None if offset is None else offset._version)
        cached = getattr(self, "_zero_joints_cache", None)
        if cached is not None and cached[0][0] is offset and cached[0][1] == key[1]:
            return cached[1].clone()
        with torch.no_grad():
            posed, _ = self._chain_uncached(torch.zeros_like(self.betas), torch.zeros_like(self.body_pose), torch.zeros_like(self.global_orient))
            if self.mesh_offset is not None:
                posed = posed + self.mesh_offset
            joints = posed[0, self._smplx_indices]
        self._zero_joints_cache = (key, joints)
        return joints.clone()

    def _working_factor(self, points):
        """WORK_SCALE over the scale this person's torso height implies."""
        if WORK_SCALE is None:
            return 1.0
        targets, weights = landmarks_to_targets(points)
        torso = torch.tensor([1.0 if index in _TORSO else 0.0 for index in self._joint_order])
        torso_weights = torch.tensor(weights, dtype=torch.float32) * torso
        estimate = self._initial_scale(
            torch.tensor(targets, dtype=torch.float32), torso_weights, fallback=0.0, rest_joints=self._zero_pose_joints()
        )
        return WORK_SCALE / estimate if estimate > 0 else 1.0

    def _rescale_camera(self, factor):
        with torch.no_grad():
            self.scale.mul_(factor)
            self.translation.mul_(factor)

    def _fit_working(self, points, iterations, learning_rate, camera_iterations, orientation_hypotheses, prefer_yaw=None):
        """Cold start, in two stages - the order SMPLify uses, and for a reason.

        Solving camera and pose together from a rest-pose start finds the
        wrong basin. Measured against a synthetic upright body with exact 3D
        ground truth: the joint fit reached 13 px in 2D while the body was
        355 mm off in 3D and tipped 28 degrees toward the camera. Under weak
        perspective a leaning body and an upright one project nearly alike,
        so once the optimiser was in that basin the 2D loss had no reason to
        leave it. Neither a stronger orientation prior nor a smaller step
        fixed it - smaller steps were worse.

        Started from the true parameters the same optimiser converges, so
        the fault is the start. SMPLify's answer (Bogo et al. 2016, Sec. 3):

        1. fit the CAMERA only, to the torso joints only, with the pose held
           at rest. The torso is rigid, so scale and translation are
           well-determined by it alone: this recovers scale 178 against a
           truth of 180.
        2. release everything, from that camera.

        Result on the same body: 0.7 px, 44 mm, 1 degree.

        MediaPipe labels joints anatomically whatever way the person faces,
        so a back view puts the left shoulder on the left of the image, and
        only a yaw near 180 degrees reproduces that. Gradient descent from a
        camera-facing start will not cross the 90-degree barrier on its own
        (it stalled at 32 degrees on the real back camera), so both yaw
        hypotheses are tried and the better kept - SMPLify's own trick for
        ambiguous initialisations. This has to include stage 1: fitting the
        camera once at yaw=0 and reusing it for the yaw=pi candidate was
        measured to fail on a real annotated frame where the true yaw is
        178 degrees - torso loss floored at 10,953 px^2 because a
        180-degree-turned torso is not a scaled, translated copy of a
        forward-facing one, so stage 2 started from a camera that did not
        fit either facing and never recovered (72 px reprojection, betas
        pinned at the prior's edge trying to explain it as a body shape).
        Fitting the camera separately per yaw candidate reaches 2.9 px on
        the same frame.

        `orientation_hypotheses`: yaw candidates to try, defaulting to
        `(0.0, pi)` (the behaviour above, unchanged for every existing
        caller). This only resolves the discrete front/back case - a
        person near-perpendicular to the camera (~45-135 degrees) has no
        seed near either hypothesis, and gradient descent from a
        camera-facing start can converge to a locally-plausible but
        wrong yaw there too (measured on real annotated frames: camera
        0076's true yaw is 65.1 degrees, landed at 197.5). the orientation classifier's
        `tryon.body.orientation.OrientationClassifier`, trained from
        scratch on synthetic data (see `tryon/body/orientation_data.py`),
        classifies which yaw octant the 2D keypoints suggest and can
        supply better seeds here than the two hardcoded extremes -
        callers that want this pass a wider set of hypotheses (e.g. the
        classifier's top-2 predicted bin centres) instead of relying on
        this default.
        """
        if orientation_hypotheses is None:
            # Facing the camera, away, and the two profiles: with only the
            # front/back pair a wearer seen in profile converged 35 degrees
            # short of sideways (measured on the annotated profile cameras).
            orientation_hypotheses = (0.0, float(np.pi), float(np.pi / 2), float(-np.pi / 2))
        targets, weights = landmarks_to_targets(points)
        self._set_depths(points)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        self._shape_history = []
        self.shape_locked = False
        visible = weights > 0
        centre = (
            torch.tensor(targets[visible].mean(axis=0), dtype=torch.float32)
            if visible.any()
            else self.translation.detach().clone()
        )
        torso = torch.tensor(
            [1.0 if index in _TORSO else 0.0 for index in self._joint_order]
        )
        torso_weights = weight_tensor * torso

        if prefer_yaw is not None:
            orientation_hypotheses = tuple(orientation_hypotheses) + (float(prefer_yaw),)
        candidates = []
        for yaw in orientation_hypotheses:
            with torch.no_grad():
                self.body_pose.zero_()
                self.betas.copy_(self._shape_mean())
                self.global_orient.zero_()
                self.global_orient[0, 1] = yaw
                self.scale.copy_(torch.tensor([self._initial_scale(target_tensor, torso_weights)]))
                self.translation.copy_(centre)
            self._optimiser = None
            self._fit_camera(target_tensor, weight_tensor, torso_weights, camera_iterations)
            loss = 0.0
            for _ in range(max(int(iterations), 1)):
                loss = self._step(target_tensor, weight_tensor, learning_rate, shape=True)
            state = [parameter.detach().clone() for parameter in self._parameters()]
            candidates.append((loss, state, float(self.global_orient.detach()[0, 1])))
        lowest = min(c[0] for c in candidates)
        best = min(candidates, key=lambda c: c[0])
        if prefer_yaw is not None:
            # In profile the loss cannot tell the facings apart (22 vs 69
            # degrees both reprojected at ~8 px on the turn clip), so among
            # the candidates within YAW_LOSS_SLACK of the best loss the one
            # nearest MediaPipe's own torso yaw wins.
            eligible = [c for c in candidates if c[0] <= lowest * YAW_LOSS_SLACK + 1e-6]
            best = min(eligible, key=lambda c: yaw_difference(c[2], prefer_yaw))
        with torch.no_grad():
            for parameter, value in zip(self._parameters(), best[1]):
                parameter.copy_(value)
        self._optimiser = None
        return best[0]

    def track(self, points, iterations=3, learning_rate=0.02):
        """Warm start from the current pose - the per-frame path.

        Shape is free while evidence accumulates, then held. An earlier
        version froze shape at the cold fit's single-frame guess and it
        bought nothing (five seeds, acceleration ratio 1.03); freezing a
        guess is wrong. Freezing an ESTIMATE - the mean over the warm-up
        window - is what gives one consistent body, at a measured cost of
        about 1 px of reprojection.
        """
        # Same normalised frame as the cold fit that started this track.
        factor = getattr(self, "_work_factor", 1.0)
        points = _scaled_points(points, factor)
        targets, weights = landmarks_to_targets(points)
        self._set_depths(points)
        self._rescale_camera(factor)
        if RECENTRE_TRACK:
            self._recentre(targets, weights)
        try:
            loss = self._run(targets, weights, iterations, learning_rate, shape=not self.shape_locked) / factor**2
        finally:
            self._rescale_camera(1.0 / factor)
        if not self.shape_locked:
            self._shape_history.append(self.betas.detach().numpy().ravel().copy())
            if len(self._shape_history) >= self.shape_frames:
                with torch.no_grad():
                    self.betas.copy_(
                        torch.tensor(np.mean(self._shape_history, axis=0), dtype=torch.float32)[None]
                    )
                self.shape_locked = True
                self._optimiser = None
        return loss

    def _recentre(self, targets, weights):
        """Move the camera translation by the mean torso residual.

        Adam moves the translation by about its step size per iteration,
        so 15 iterations let the body follow at ~1 px a frame; a dancer
        moving faster left the fit behind and the pose contorted to
        compensate (00176_outer: reprojection drifting 8 -> 38 px until a
        refit). The hips' mean offset is the translation error, applied in
        closed form before the iterations. Hips only, not the whole torso:
        under weak perspective a turn left and a turn right narrow the
        torso alike, and the sign lives in the shoulders' sideways shift
        against the hips - recentring on all four erased it (the 90-frame
        synthetic turn ended at +63 instead of -68 degrees). The hip
        midpoint sits on the pelvis and does not move with yaw.
        """
        hips = (_IDS.LEFT_HIP.value, _IDS.RIGHT_HIP.value)
        torso = np.array([1.0 if index in hips else 0.0 for index in self._joint_order]) * weights
        if torso.sum() <= 0:
            return
        with torch.no_grad():
            projected = self._project(self.joints()).numpy()
            shift = ((targets - projected) * torso[:, None]).sum(axis=0) / torso.sum()
            # Only the part beyond a dead zone: the few-px offsets that
            # carry the yaw sign stay for the optimiser to explain by
            # turning; a fast move (tens of px) is taken up at once.
            dead = RECENTRE_DEAD_ZONE * float(self.scale.detach())
            shift = shift - np.clip(shift, -dead, dead)
            self.translation.add_(torch.tensor(shift, dtype=torch.float32)[None])

    def _set_depths(self, points):
        depths, depth_weights = landmarks_to_depths(points)
        self._depth_targets = torch.tensor(depths, dtype=torch.float32)
        self._depth_weights = torch.tensor(depth_weights, dtype=torch.float32)

    def _run(self, targets, weights, iterations, learning_rate, shape=True):
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        loss = 0.0
        count = max(int(iterations), 1)
        for index in range(count):
            # the silhouette term (see `set_silhouette`) skins the whole mesh,
            # ~7.5 ms an iteration, so it runs on the last few only
            self._silhouette_active = (self._silhouette is not None or self._skin_points is not None) and index >= count - SILHOUETTE_ITERATIONS
            loss = self._step(target_tensor, weight_tensor, learning_rate, shape)
        self._silhouette_active = False
        return loss

    _silhouette = None
    _silhouette_active = False

    _skin_points = None

    def set_skin(self, skin_mask, count=400):
        """Pixels of the wearer's exposed skin (face, arms, hands, legs) for
        the next `track`, or None: each sampled skin pixel should have a
        projected body vertex near it (SKIN_WEIGHT). Two-sided where it is
        safe - skin is always body, unlike clothing."""
        if skin_mask is None or SKIN_WEIGHT <= 0:
            self._skin_points = None
            return
        ys, xs = np.nonzero(np.asarray(skin_mask, bool))
        if len(xs) < 20:
            self._skin_points = None
            return
        pick = np.random.default_rng(len(xs)).choice(len(xs), min(count, len(xs)), replace=False)
        self._skin_points = torch.tensor(np.stack([xs[pick], ys[pick]], axis=1), dtype=torch.float32)

    def _skin_loss(self):
        """Mean squared distance (working px) from sampled skin pixels to the
        nearest projected body vertex."""
        if getattr(self, "_silhouette_vertices", None) is None:
            rng = np.random.default_rng(0)
            self._silhouette_vertices = torch.as_tensor(np.sort(rng.choice(10475, 1500, replace=False)))
        vertices = self._forward().vertices[0][self._silhouette_vertices]
        factor = getattr(self, "_work_factor", 1.0)
        pixels = self._project(vertices) / factor
        nearest = torch.cdist(self._skin_points, pixels).min(dim=1).values
        return ((nearest * factor) ** 2).mean()

    def set_silhouette(self, person_mask):
        """The person's silhouette (bool, frame pixels) for the next `track`,
        or None. Body vertices projecting OUTSIDE it are pulled in by
        SILHOUETTE_WEIGHT x their squared distance to it (a distance
        transform sampled bilinearly, so it has a gradient). One-sided on
        purpose: loose clothing widens the silhouette, and a body pushed out
        to fill it would be wrong; a body sticking out past the person is
        wrong whatever they wear. The authors' registration fits the image;
        ours matched the person's silhouette at 0.70 IoU against its 0.90,
        and our garment's extra pixels fell mostly on the background."""
        if person_mask is None or SILHOUETTE_WEIGHT <= 0:
            self._silhouette = None
            return
        mask = np.asarray(person_mask, bool)
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        self._silhouette = torch.from_numpy(outside)[None, None]
        if getattr(self, "_silhouette_vertices", None) is None:
            rng = np.random.default_rng(0)
            self._silhouette_vertices = torch.as_tensor(np.sort(rng.choice(10475, 1500, replace=False)))

    def _silhouette_loss(self):
        """Mean squared distance (working px) of sampled body vertices outside
        the silhouette."""
        vertices = self._forward().vertices[0][self._silhouette_vertices]
        factor = getattr(self, "_work_factor", 1.0)
        pixels = self._project(vertices) / factor  # frame pixels
        height, width = self._silhouette.shape[-2:]
        grid = torch.stack([pixels[:, 0] / (width - 1) * 2 - 1, pixels[:, 1] / (height - 1) * 2 - 1], dim=1)[None, None]
        distance = torch.nn.functional.grid_sample(self._silhouette, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return ((distance.reshape(-1) * factor) ** 2).mean()

    def reprojection_error(self, points):
        """Mean pixel distance over the landmarks actually being fitted."""
        targets, weights = landmarks_to_targets(points)
        with torch.no_grad():
            projected = self._project(self.joints()).numpy()
        distance = np.linalg.norm(projected - targets, axis=1)
        total = weights.sum()
        return float((distance * weights).sum() / total) if total else float("inf")

    def orientation_hypotheses_from_regressor(self, points, regressor, top_k=2):
        """Yaw seeds for `fit`'s `orientation_hypotheses`, from the orientation classifier's
        `OrientationClassifier` instead of the hardcoded {0, pi} - see
        `fit`'s docstring for why the hardcoded pair misses the
        near-perpendicular case. Falls back to (0, pi) if `points` has no
        usable landmarks (matches `fit`'s own no-visible-landmark
        fallback).
        """
        import torch as _torch

        from tryon.body.orientation_data import _normalize_features, bin_center

        targets, weights = landmarks_to_targets(points)
        visible = weights > 0
        if not visible.any():
            return (0.0, float(np.pi))

        centre = targets[visible].mean(axis=0)
        # Torso scale: same quantity `fit`'s stage-1 camera fit solves
        # for, approximated here directly from the observed keypoints
        # (shoulder-hip spread) since the regressor needs a normalisation
        # scale before any fit has run yet.
        spread = float(np.linalg.norm(targets[visible].max(axis=0) - targets[visible].min(axis=0)))
        scale = max(spread, 1.0)
        visibility = (weights > 0).astype(np.float64)
        features = _normalize_features(targets, visibility, scale, centre)

        with _torch.no_grad():
            logits = regressor(_torch.as_tensor(features, dtype=_torch.float32).unsqueeze(0))
            top_bins = _torch.topk(logits[0], k=min(top_k, logits.shape[-1])).indices.tolist()
        return tuple(bin_center(bin_index) for bin_index in top_bins)

    def orientation_disagrees(self, points, regressor, threshold_deg=90.0):
        """Whether the regressor's independently-predicted yaw octant
        disagrees with the currently tracked `global_orient` by more than
        `threshold_deg` - a cheap (one small-MLP forward pass, no
        gradient) periodic sanity check for a `track()`-warm-started fit
        that may have gotten stuck (see `fit`'s docstring on why gradient
        descent can fail to cross a large orientation gap on its own).
        """
        hypotheses = self.orientation_hypotheses_from_regressor(points, regressor, top_k=1)
        predicted_yaw = hypotheses[0]
        with torch.no_grad():
            current_yaw = float(self.global_orient[0, 1])
        difference = abs(((predicted_yaw - current_yaw) + np.pi) % (2 * np.pi) - np.pi)
        return np.degrees(difference) > threshold_deg

    def vertices(self):
        """The posed body mesh, in model space."""
        with torch.no_grad():
            return self._forward().vertices[0].numpy()

    def projected_vertices(self):
        """The posed body mesh in image pixels, for compositing."""
        with torch.no_grad():
            vertices = self._forward().vertices[0]
            return self._project(vertices).numpy()

    @property
    def faces(self):
        return self.model.faces
