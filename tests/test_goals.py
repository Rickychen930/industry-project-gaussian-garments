"""The two goals, tested on the real assets rather than asserted.

These are the properties the 2D system could only approximate: it faked
rotation with 24 pre-rendered views and faked motion by warping anchors,
and it hit a wall at the sleeves because an outstretched arm needs the
upper arm turned -72 degrees while the torso stays at 0, and one affine
transform carries one rotation.

A garment skinned to a body gets both for free. These tests are what stops
that claim from being a claim.
"""

from __future__ import annotations

import unittest

import numpy as np

from tryon.paths import SMPLX_ROOT, garment

GARMENT = garment("00152_outer") / "stage1" / "template_uv.obj"
# SMPL-X body_pose is 21 joints x 3; shoulders are joints 15 and 16.
LEFT_SHOULDER_Z = 15 * 3 + 2
RIGHT_SHOULDER_Z = 16 * 3 + 2


def _setup():
    import smplx
    import torch
    import trimesh

    from tryon.garment.skinning import GarmentBinding, align_to_body

    model = smplx.create(
        str(SMPLX_ROOT), model_type="smplx", gender="neutral",
        use_pca=False, batch_size=1,
    )
    with torch.no_grad():
        rest = model(
            betas=torch.zeros(1, 10),
            body_pose=torch.zeros(1, 63),
            global_orient=torch.zeros(1, 3),
        ).vertices[0].numpy()
    mesh = trimesh.load(str(GARMENT), process=False)
    garment = align_to_body(np.asarray(mesh.vertices), rest)
    return model, rest, garment, GarmentBinding.build(garment, rest)


def _posed(model, body_pose=None, yaw=0.0):
    import torch

    orient = torch.zeros(1, 3)
    orient[0, 1] = float(yaw)
    with torch.no_grad():
        return model(
            betas=torch.zeros(1, 10),
            body_pose=body_pose if body_pose is not None else torch.zeros(1, 63),
            global_orient=orient,
        ).vertices[0].numpy()


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X models at {SMPLX_ROOT}")
@unittest.skipUnless(GARMENT.exists(), f"no garment mesh at {GARMENT}")
class ThreeDGoalsTest(unittest.TestCase):
    def test_the_garment_turns_with_the_body_through_a_full_circle(self):
        """360 degrees, structurally - no view catalog, no morphing."""
        model, rest, garment, binding = _setup()
        centre = rest.mean(axis=0)
        for degrees in range(0, 360, 45):
            radians = np.radians(degrees)
            posed = binding.apply(_posed(model, yaw=radians))
            rotation = np.array(
                [
                    [np.cos(radians), 0, np.sin(radians)],
                    [0, 1, 0],
                    [-np.sin(radians), 0, np.cos(radians)],
                ]
            )
            ideal = (garment - centre) @ rotation.T + centre
            deviation = float(np.median(np.linalg.norm(posed - ideal, axis=1)))
            self.assertLess(
                deviation,
                0.05,
                f"at {degrees} deg the garment deviated {deviation:.3f} m from "
                "a rigid turn of itself",
            )

    def test_a_profile_view_is_narrower_than_a_head_on_one(self):
        """The evidence that it really turned, rather than staying flat."""
        model, _rest, _garment, binding = _setup()
        front = binding.apply(_posed(model, yaw=0.0))
        side = binding.apply(_posed(model, yaw=np.radians(90)))
        front_width = front[:, 0].max() - front[:, 0].min()
        side_width = side[:, 0].max() - side[:, 0].min()
        self.assertLess(
            side_width,
            front_width * 0.8,
            f"profile {side_width:.3f} m against head-on {front_width:.3f} m",
        )

    def test_the_sleeves_follow_the_arms(self):
        """The -72 degree limb rotation the 2D warp could not express."""
        import torch

        model, _rest, _garment, binding = _setup()
        down = binding.apply(_posed(model))
        raised = torch.zeros(1, 63)
        raised[0, LEFT_SHOULDER_Z] = -1.2
        raised[0, RIGHT_SHOULDER_Z] = 1.2
        out = binding.apply(_posed(model, body_pose=raised))
        moved = np.linalg.norm(out - down, axis=1)
        # The sleeves must move a long way while the torso stays put: that
        # difference is the whole point.
        self.assertGreater(
            float(np.percentile(moved, 95)),
            0.10,
            "raising the arms barely moved the garment",
        )
        self.assertLess(
            float(np.median(moved)),
            float(np.percentile(moved, 95)) * 0.6,
            "the whole garment moved, so it is not following the limbs",
        )

    def test_the_garment_stays_on_the_body(self):
        """Accuracy: it must sit ON the wearer, not float beside them."""
        from scipy.spatial import cKDTree

        model, _rest, _garment, binding = _setup()
        for degrees in (0, 90, 180, 270):
            body = _posed(model, yaw=np.radians(degrees))
            worn = binding.apply(body)
            distance, _ = cKDTree(body).query(worn)
            self.assertLess(
                float(np.median(distance)),
                0.12,
                f"at {degrees} deg the garment sat {np.median(distance):.3f} m "
                "from the body surface",
            )


if __name__ == "__main__":
    unittest.main()
