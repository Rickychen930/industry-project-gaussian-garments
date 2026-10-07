"""open3d ray casting against triangle meshes: the scene construction and the
camera-to-splat occlusion test the garment code shares."""

from __future__ import annotations

import numpy as np


def build_scene(meshes):
    """An open3d `RaycastingScene` holding every (vertices, faces) in `meshes`
    (float32 positions, int32 triangle indices)."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    for vertices, faces in meshes:
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(np.asarray(vertices, np.float32))
        mesh.triangle.indices = o3d.core.Tensor(np.ascontiguousarray(faces, np.int32))
        scene.add_triangles(mesh)
    return scene


def hit_distance(scene, origins, directions):
    """Distance along each ray (unit `directions`) to its first hit, inf when it misses."""
    import open3d as o3d

    rays = np.ascontiguousarray(np.concatenate([origins, directions], axis=1), np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


def occluded_from(xyz, camera, body_vertices, body_faces, margin):
    """Which of `xyz` the body hides from a camera at `camera`: a ray from the
    camera to the point hits the body more than `margin` metres before
    reaching it (`GarmentRenderer.occluded_by_body`)."""
    scene = build_scene([(body_vertices, body_faces)])
    xyz = np.asarray(xyz, np.float32)
    camera = np.asarray(camera, np.float32)
    direction = xyz - camera
    distance = np.linalg.norm(direction, axis=1)
    direction /= np.maximum(distance, 1e-9)[:, None]
    first_hit = hit_distance(scene, np.broadcast_to(camera, xyz.shape), direction)
    return first_hit < distance - margin
