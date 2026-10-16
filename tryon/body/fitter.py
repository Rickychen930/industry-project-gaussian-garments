"""Fit a SMPL-X body to MediaPipe landmarks, then track it.

SMPLify (Bogo et al. 2016; Kolotouros et al. arXiv:1909.12828): minimise the distance
between projected model joints and detected 2D keypoints, with a prior keeping the pose
plausible. A body made of joints can express limb rotations a single 2D affine cannot.

Design constraints:

* SMPL-X is 6 ms on CPU against 39 ms on MPS (padding falls back to a view op), so this
  module stays on CPU.
* A cold fit (30 iterations) costs ~420 ms; warm-started, 3 iterations cost ~42 ms. So
  the class is stateful: `fit` once when a person appears, `track` every frame after.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

# Straight from MediaPipe, not `tryon.views`: the 3D path must not depend on the 2D stack.
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

# Weight of the bone-depth term relative to the 2D term. The depth residual (metres)
# is scaled by the camera's px/m so equal errors in depth and in-plane cost the same,
# times this factor.
DEPTH_WEIGHT = 0.5
# Camera scale (px per metre) the fit runs at: landmarks are rescaled so the torso
# implies this scale, fitted, and the camera scaled back. None fits in the caller's
# pixels. See `BodyFitter.fit`.
WORK_SCALE = 120.0
# Use MediaPipe's world-landmark torso yaw (`landmark_yaw`) as a facing hypothesis and
# tie-breaker, and a refit trigger in live.track_step.
USE_LANDMARK_YAW = True
# Silhouette term in `track` (see BodyFitter.set_silhouette): weight, and how many of
# the last iterations of a frame carry it. 0 = off. Weight 3 raised IoU against the
# authors' registration on most clips for ~25 ms a frame.
SILHOUETTE_WEIGHT = 3.0
SILHOUETTE_ITERATIONS = 3
# Candidates within this factor of the best loss are "equally good"; the one facing
# MediaPipe's way is taken.
YAW_LOSS_SLACK = 1.3
# Keypoint refinement steps after an image regressor's seed (`BodyFitter.fit_from_regression`).
REGRESSION_ITERATIONS = 50
# Where a regressed body is placed in the image: "regressor" (its own
# projected joints) or "keypoints" (camera fitted to MediaPipe).
REGRESSION_CAMERA = "regressor"
# `track` shifts the camera by the hips' mean residual beyond a dead zone before
# iterating (see `BodyFitter._recentre`). Without the dead zone, or with all four torso
# joints, a turn is mirrored.
RECENTRE_TRACK = True
# metres of hip offset left to the optimiser (see `_recentre`)
RECENTRE_DEAD_ZONE = 0.02


_SMPLX_MODELS = {}  # (models root, gender) -> the loaded model, copied per BodyFitter


class BodyFitter:
    """A SMPL-X body that is fitted once and then tracked."""

    # Weight of the shape prior in the 2D fit. Keypoints constrain bone lengths, never
    # girth, so a weak prior lets the shape drift wide and a garment bound to the scan
    # body ends up inside the live one. Stronger priors fit the true body better but raise
    # profile-view reprojection past the real-photo test threshold; 400 is the compromise.
    SHAPE_PRIOR = 400.0
    # Arm-only refinement of a regressed pose toward MediaPipe's 2D elbows and wrists
    # (`_refine_arms`). At a gate of 0.12 torso lengths it touched most arms and lowered
    # garment IoU; at 0.4 it acts only where NLF is badly wrong (an arm toward the camera).
    ARM_REFINE = True
    ARM_GATE = 0.4  # torso lengths of elbow / wrist disagreement before an arm is touched
    # Torso length (shoulder to hip) in shoulder widths, for a wearer whose hips are out
    # of frame (the unit the gate is counted in).
    TORSO_PER_SHOULDER_WIDTH = 1.3
    ARM_VISIBILITY = 0.5  # MediaPipe visibility of a landmark worth following
    ARM_BLEND = 1.0  # how far toward MediaPipe's point the target sits (1: on it)
    ARM_PRIORS = (0.3, 1.0, 0.1)  # per rad: shoulder, elbow non-hinge, elbow flexion
    POSE_PRIOR = 60.0
    # Centre of the pose prior (1, 63): None = rest; `fit_from_regression` sets it to the
    # regressor's pose so the keypoint term refines it instead of pulling toward a T-pose.
    pose_anchor = None
    # Centre of the shape prior (10 betas); the fit collapses onto it at WORK_SCALE, so it
    # is the shape a monocular fit ends up with. None = SMPL-X's mean (zeros).
    shape_mean = None

    def _shape_mean(self):
        if self.shape_mean is None:
            return torch.zeros_like(self.betas)
        return torch.as_tensor(np.asarray(self.shape_mean, np.float32)).reshape(self.betas.shape)

    def __init__(self, gender="neutral", models_root=SMPLX_ROOT, threads=8, shape_frames=30):
        import smplx

        torch.set_num_threads(threads)
        # The model file is read once per process (~0.5-1.5 s); every fitter gets a deep copy.
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
        # Weak perspective: a scale and a 2D offset, as SMPLify uses for an unknown camera.
        self.scale = torch.tensor([250.0], requires_grad=True)
        self.translation = torch.zeros(1, 2, requires_grad=True)
        self._joint_order = sorted(JOINT_MAP)
        self._smplx_indices = [JOINT_MAP[index] for index in self._joint_order]
        # Bone-depth targets for the current frame (see landmarks_to_depths),
        # set by fit()/track(); None = no depth term.
        self._depth_pairs = [(self._joint_order.index(a), self._joint_order.index(b)) for a, b in _all_depth_bones()]
        self._depth_targets = None
        self._depth_weights = None
        # The 20-wide blend basis the library uses (`model.shapedirs` is the 10-wide slice).
        self._shapedirs_full = torch.cat([self.model.shapedirs, self.model.expr_dirs], dim=-1)
        self._optimiser = None
        self._optimiser_key = None
        # Shape is estimated over the first `shape_frames` tracked frames and then held:
        # skeleton keypoints carry almost no shape information, so per-frame shape is noise.
        self.shape_frames = int(shape_frames)
        self._shape_history = []
        self.shape_locked = False

    # A 3D world offset (1, 3) applied to vertices and joints, set by `fit_mesh` (and by
    # GarmentRenderer for the scan-frame body); None for the live 2D camera fit.
    mesh_offset = None

    def _forward(self):
        """The library's full forward: the skinned mesh. Used for vertices.

        Without autograd the result is cached until a parameter changes (a live frame asks
        for the same posed body several times). Callers get fresh tensors."""
        if torch.is_grad_enabled():
            return self._forward_uncached()
        # the tensors themselves are kept (not ids) so a replaced parameter cannot alias a freed one
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
        """Fit shape, pose and orientation to a SMPL-X-topology mesh (10,475 vertices) in 3D,
        such as the scan-frame body in the Gaussian-Garments trajectory files. Returns the
        mean vertex residual in metres. The 3D offset between model and target is solved as
        `self.mesh_offset`."""
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

        # Rigid first (orientation + offset), then everything: solving all at once from
        # the rest pose left the joints chasing an unsettled orientation.
        run([self.global_orient, self.mesh_offset], max(int(iterations) // 4, 50), learning_rate)
        run([self.betas, self.body_pose, self.global_orient, self.mesh_offset], iterations, learning_rate)
        self._optimiser = None
        return residual

    def joints(self, indices=None):
        """The fitted joints without skinning the mesh - the tracking path.

        The library regresses joints from the shaped template before skinning; doing only
        that is ~4x faster and matches to 1e-7 m. Two inputs must match what the library
        actually passes to `lbs`:

        * shape is blended through a 20-wide `shapedirs` (betas plus expression), not the
          10-wide `model.shapedirs`.
        * hands are posed at `left_hand_mean` / `right_hand_mean`, not zero (zero bends the
          wrists 4 cm, exactly where a sleeve hangs).
        """
        posed, _ = self._chain()
        if self.mesh_offset is not None:
            posed = posed + self.mesh_offset
        return posed[0, self._smplx_indices if indices is None else indices]

    def _chain(self):
        """(posed joints, joint transforms) before the world offset: rest joints from the
        shaped template, then the kinematic chain.

        Without autograd the result is cached until a parameter changes (like `_forward`).
        The cached tensors are never handed out: callers add the offset into new tensors."""
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

        Skinning applies them: a vertex with weights `w` moves by `sum_j w_j A_j`; both the
        body model and `bind.SkinnedGarment` use them. Same chain as `joints()`.
        """
        _, transforms = self._chain()
        transforms = transforms[0].clone()  # never a view of the cache
        if self.mesh_offset is not None:
            # `_forward` adds the world offset AFTER skinning; weights sum to one, so adding it
            # to every joint's translation is the same and a garment skinned by these transforms
            # lands in the frame of `vertices()`.
            transforms[:, :3, 3] = transforms[:, :3, 3] + self.mesh_offset
        return transforms

    def _project(self, joints):
        """Weak-perspective projection: image y grows downward, model y up."""
        flipped = joints[..., :2] * torch.tensor([1.0, -1.0])
        return flipped * self.scale + self.translation

    def _parameters(self, shape=True):
        """What the optimiser may move. Shape only during the cold fit.

        A body does not change shape between frames, so `track` leaves beta out of the
        optimiser: it cannot drift or explain a pose change as a shape change.
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
            # Bone depth from MediaPipe's 3D estimate (DEPTH_BONES), in pixel-equivalent units.
            a = torch.tensor([p[0] for p in self._depth_pairs])
            b = torch.tensor([p[1] for p in self._depth_pairs])
            dz = joints[b, 2] - joints[a, 2]
            depth = ((dz - self._depth_targets) ** 2 * self._depth_weights).sum() / self._depth_weights.sum()
            data = data + DEPTH_WEIGHT * self.scale.detach() ** 2 * depth
        # Keep the pose near rest and the shape near average, or a 63-dimensional pose
        # contorts to hit 13 noisy keypoints.
        #
        # The global orientation needs its own term: under weak perspective a body tipped
        # toward the camera projects almost like an upright one, so the 2D loss is indifferent
        # (a fit leaned 65 degrees with 12.9 px reprojection). Pitch and roll only: yaw is
        # free (anchoring it stalled the back camera halfway round).
        upright = self.global_orient[:, [0, 2]]
        anchor = self.pose_anchor if self.pose_anchor is not None else 0.0
        prior = (
            self.POSE_PRIOR * ((self.body_pose - anchor) ** 2).mean()
            + self.SHAPE_PRIOR * ((self.betas - self._shape_mean()) ** 2).mean()
            + 400.0 * (upright**2).mean()
        )
        if self._silhouette_active and SILHOUETTE_WEIGHT > 0 and self._silhouette is not None:
            prior = prior + SILHOUETTE_WEIGHT * self._silhouette_loss()
        loss = data + prior
        loss.backward()
        self._optimiser.step()
        return float(data.detach())

    def _initial_scale(self, target_tensor, torso_weights, fallback=250.0, rest_joints=None):
        """Camera scale seed: observed torso height over the rest torso's.

        A fixed seed cannot be moved far by the camera stage. Vertical shoulder-to-hip extent
        is used because, unlike shoulder width, it does not shrink in profile. Called with
        the pose at rest and the yaw hypothesis set. `rest_joints` (precomputed joints of
        that rest pose) skips the kinematic chain.
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

        With the CORRECT facing the rigid torso is well determined by four points; with the
        WRONG facing no scale and translation reproduce it and the loss floors high. Called
        once per yaw hypothesis in `fit`, so each stage 2 candidate starts from a camera
        consistent with its own facing.
        """
        camera = torch.optim.Adam([self.scale, self.translation], 2.0)
        for _ in range(max(int(camera_iterations), 1)):
            camera.zero_grad()
            residual = ((self._project(self.joints()) - target_tensor) ** 2).sum(dim=1)
            (residual * torso_weights).sum().div(torso_weights.sum().clamp(min=1e-6)).backward()
            camera.step()

    def fit(self, points, iterations=200, learning_rate=0.02, camera_iterations=100, orientation_hypotheses=None, prefer_yaw="landmarks"):
        """Cold fit in a normalised frame - see `_fit_working` for the method.

        The landmarks are rescaled so the camera scale is WORK_SCALE px/m, fitted there, and
        scaled back: fitting at the full panel size scored a worse silhouette IoU than at a
        small working size, and MediaPipe's input size made no difference. The returned loss
        is px^2 of the caller's image.
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

        Pose, shape and facing come from the regressor; only the camera (torso, stage 1 of
        `_fit_working`) and a short keypoint refinement come from MediaPipe, with the pose
        prior centred on the regressed pose (`pose_anchor`). `prediction`: dict with
        global_orient (3,), body_pose (63,), betas (10,) and, for `camera` "regressor",
        joints_2d (22, 2). `camera`: None = REGRESSION_CAMERA.
        """
        camera = REGRESSION_CAMERA if camera is None else camera
        landmarks = points
        if points is None:
            # MediaPipe lost the wearer (arm or object across the torso); the regressor still
            # saw them. No landmark is used.
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
            # each prediction's own betas: holding a mean of the first frames scored no better
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
            # The regressor's own placement: least-squares scale and translation of our 22 body
            # joints onto its projected ones (its camera is perspective, ours weak).
            with torch.no_grad():
                ours = self._project(self.joints(list(range(22)))).numpy()
                ours = (ours - self.translation.detach().numpy()) / float(self.scale)
            wanted = np.asarray(prediction["joints_2d"], np.float64)
            # Only joints it places inside the image: for a wearer cut off at the waist the legs
            # are guesses and fitting them put the garment too high.
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
        # Root and camera only: releasing pose or shape to the keypoints undid the regressor
        # (PA-MPJPE 50 mm seeded, 83 mm after 10 all-parameter steps).
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

        if min(landmarks[i, 2] for i in (11, 12)) < VISIBILITY_FLOOR:
            return []
        if min(landmarks[i, 2] for i in (23, 24)) >= VISIBILITY_FLOOR:
            length = 0.5 * float(np.linalg.norm(landmarks[11, :2] - landmarks[23, :2]) + np.linalg.norm(landmarks[12, :2] - landmarks[24, :2]))
        else:
            length = self.TORSO_PER_SHOULDER_WIDTH * float(np.linalg.norm(landmarks[11, :2] - landmarks[12, :2]))
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

        For a tracked body whose reprojection suddenly jumps (a fast turn, a camera cut):
        warm-started tracking turns only gradually. The current yaw is one of the
        hypotheses and the old state is restored unless the refit reprojects better.
        Returns True when adopted.
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
            # Called because the facing disagrees with MediaPipe's: adopt a refit that agrees
            # better if it reprojects about as well.
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
        """`joints()` at zero pose and zero betas, computed once: it depends only on the
        model and the (normally absent) mesh offset."""
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
        """Cold start, in two stages - the order SMPLify uses, for a reason.

        Solving camera and pose together from a rest-pose start finds the wrong basin: under
        weak perspective a leaning body and an upright one project nearly alike, so the 2D
        loss has no reason to leave it, and neither a stronger prior nor a smaller step
        fixes it. So (Bogo et al. 2016, Sec. 3):

        1. fit the CAMERA only, to the torso joints only, with the pose held at rest.
        2. release everything, from that camera.

        MediaPipe labels joints anatomically whatever way the person faces, so a back view
        needs a yaw near 180 degrees, and gradient descent from a camera-facing start will not
        cross the 90-degree barrier. Both yaw hypotheses are tried and the better kept,
        including stage 1 per candidate (a camera fitted once at yaw=0 fails for a back view).

        `orientation_hypotheses`: yaw candidates to try, default `(0.0, pi)`. That resolves
        only front/back; a near-perpendicular person has no seed near either, so callers can
        pass the orientation classifier's top bin centres
        (`tryon.body.orientation.OrientationClassifier`) instead.
        """
        if orientation_hypotheses is None:
            # Facing the camera, away, and the two profiles: front/back alone converged short of
            # sideways for a profile.
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
            # In profile the loss cannot tell the facings apart, so among the candidates within
            # YAW_LOSS_SLACK of the best loss the one nearest MediaPipe's torso yaw wins.
            eligible = [c for c in candidates if c[0] <= lowest * YAW_LOSS_SLACK + 1e-6]
            best = min(eligible, key=lambda c: yaw_difference(c[2], prefer_yaw))
        with torch.no_grad():
            for parameter, value in zip(self._parameters(), best[1]):
                parameter.copy_(value)
        self._optimiser = None
        return best[0]

    def track(self, points, iterations=3, learning_rate=0.02):
        """Warm start from the current pose - the per-frame path.

        Shape is free while evidence accumulates over the warm-up window, then held: the
        mean over the window gives one consistent body (an earlier freeze of the cold fit's
        single-frame guess was no better than none).
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
        """Move the camera translation by the mean hip residual.

        Adam moves the translation by about its step size per iteration, so a fast mover
        leaves the fit behind and the pose contorts to compensate; the hips' mean offset is
        applied in closed form first. Hips only: under weak perspective left and right turns
        narrow the torso alike and the sign lives in the shoulders' shift against the hips -
        recentring on all four erased it.
        """
        hips = (_IDS.LEFT_HIP.value, _IDS.RIGHT_HIP.value)
        torso = np.array([1.0 if index in hips else 0.0 for index in self._joint_order]) * weights
        if torso.sum() <= 0:
            return
        with torch.no_grad():
            projected = self._project(self.joints()).numpy()
            shift = ((targets - projected) * torso[:, None]).sum(axis=0) / torso.sum()
            # Only the part beyond a dead zone: few-px offsets carry the yaw sign and are left to
            # the optimiser; a fast move is taken up at once.
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
            # the silhouette term skins the whole mesh (~7.5 ms an iteration): last few only
            self._silhouette_active = self._silhouette is not None and index >= count - SILHOUETTE_ITERATIONS
            loss = self._step(target_tensor, weight_tensor, learning_rate, shape)
        self._silhouette_active = False
        return loss

    _silhouette = None
    _silhouette_active = False

    def set_silhouette(self, person_mask):
        """The person's silhouette (bool, frame pixels) for the next `track`, or None. Body
        vertices projecting OUTSIDE it are pulled in by SILHOUETTE_WEIGHT x their squared
        distance to it (a bilinearly sampled distance transform). One-sided on purpose: loose
        clothing widens the silhouette, so a body pushed out to fill it would be wrong."""
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
        """Mean squared distance (working px) of sampled body vertices outside the silhouette."""
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
        """Yaw seeds for `fit`'s `orientation_hypotheses`, from the orientation classifier
        instead of the hardcoded {0, pi}. Falls back to (0, pi) without usable landmarks.
        """
        import torch as _torch

        from tryon.body.orientation_data import _normalize_features, bin_center

        targets, weights = landmarks_to_targets(points)
        visible = weights > 0
        if not visible.any():
            return (0.0, float(np.pi))

        centre = targets[visible].mean(axis=0)
        # Torso scale from the observed keypoints (shoulder-hip spread): the regressor needs
        # a normalisation scale before any fit has run.
        spread = float(np.linalg.norm(targets[visible].max(axis=0) - targets[visible].min(axis=0)))
        scale = max(spread, 1.0)
        visibility = (weights > 0).astype(np.float64)
        features = _normalize_features(targets, visibility, scale, centre)

        with _torch.no_grad():
            logits = regressor(_torch.as_tensor(features, dtype=_torch.float32).unsqueeze(0))
            top_bins = _torch.topk(logits[0], k=min(top_k, logits.shape[-1])).indices.tolist()
        return tuple(bin_center(bin_index) for bin_index in top_bins)

    def orientation_disagrees(self, points, regressor, threshold_deg=90.0):
        """Whether the classifier's independent yaw octant disagrees with the tracked
        `global_orient` by more than `threshold_deg`: a cheap periodic check for a warm
        track that got stuck.
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
