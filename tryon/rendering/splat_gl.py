"""The Gaussian compositing of `splat_cpu.render_splats`, on the GPU (moderngl).

The per-splat pre-pass is `splat_cpu._prepare`, shared unchanged. Each splat
is one instanced quad over the CPU path's pixel box, drawn farthest first;
the fragment evaluates the same Gaussian and blends premultiplied colour
ONE, ONE_MINUS_SRC_ALPHA into a float32 target. Matches the CPU to float32
rounding. The CPU sorts per tile, this sorts once; same order wherever
splats overlap.

    image, alpha = render_splats_gl(splats, view, intrinsics, (height, width))
"""

from __future__ import annotations

import numpy as np
from numba import njit

from tryon.rendering.splat_cpu import _prepare, camera_coordinates


@njit(cache=True)
def _instances(out, order, x0, y0, x1, y1, pixels, cov2d, determinant, colours, opacity, xyz, rotation, translation, with_distance):
    """Fill `out[:len(order)]` with per-instance float32 attribute rows in draw order.

    Column 14 is the camera -> splat distance for the body-occlusion test (0
    without an occluder), computed to match numpy bit for bit; column 15 is
    padding. Serial so it never overlaps another thread's numba parallel region."""
    for i in range(len(order)):
        k = order[i]
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = x0[k], y0[k], x1[k], y1[k]
        out[i, 4], out[i, 5] = pixels[k, 0], pixels[k, 1]
        out[i, 6], out[i, 7], out[i, 8] = cov2d[k, 0, 0], cov2d[k, 0, 1], cov2d[k, 1, 1]
        out[i, 9] = determinant[k]
        out[i, 10], out[i, 11], out[i, 12] = colours[k, 2], colours[k, 1], colours[k, 0]  # BGR
        out[i, 13] = opacity[k]
        distance = 0.0
        if with_distance:
            cx, cy, cz = camera_coordinates(xyz[k, 0], xyz[k, 1], xyz[k, 2], rotation, translation)
            distance = np.sqrt(cx * cx + cy * cy + cz * cz)
        out[i, 14] = distance
        out[i, 15] = 0.0


def far_to_near(depth):
    """Indices ordering `depth` farthest first.

    Two stable 16-bit radix passes on a 32-bit key spanning the depth range,
    ~4x faster than `np.argsort(-depth)`; order differs only for depths closer
    than 2^-32 of the range."""
    count = depth.shape[0]
    if count < 2048:
        return np.argsort(-depth)
    low, high = float(depth.min()), float(depth.max())
    span = high - low
    if not np.isfinite(span) or span <= 0.0:
        return np.arange(count)
    key = ((high - depth) * (4294967295.0 / span)).astype(np.uint32)  # 0 at the farthest
    low_order = np.argsort((key & 0xFFFF).astype(np.uint16), kind="stable")
    return low_order[np.argsort((key[low_order] >> 16).astype(np.uint16), kind="stable")]


_INSTANCE_ROWS = []  # one entry: the (capacity, 16) float32 array each frame fills a prefix of


def _instance_rows(count):
    """A (count, 16) float32 view of one array reused across frames (avoids a fresh 5 MB allocation per frame)."""
    if not _INSTANCE_ROWS or len(_INSTANCE_ROWS[0]) < count:
        _INSTANCE_ROWS[:] = [np.empty((max(count, 1024) + count // 4, 16), np.float32)]
    return _INSTANCE_ROWS[0][:count]


_VERTEX = """
#version 330 core
uniform vec2 size;                 // (width, height) in pixels
in vec2 corner;                    // (0|1, 0|1)
in vec4 box;                       // x0, y0, x1, y1: pixel edges
in vec2 centre;
in vec4 conic;                     // a, b, c, det
in vec3 bgr;
in float opacity;
in float dist;                     // camera -> splat centre, world units
uniform sampler2D body;            // camera -> nearest body surface per pixel
uniform int use_body;
uniform float margin;
out vec2 v_centre;
out vec4 v_conic;
out vec3 v_bgr;
out float v_opacity;
void main() {
    if (use_body == 1) {
        // the pixel under the splat's centre, rows counted from the bottom
        ivec2 texel = ivec2(int(floor(centre.x + 0.5)), int(size.y) - 1 - int(floor(centre.y + 0.5)));
        if (texel.x >= 0 && texel.y >= 0 && texel.x < int(size.x) && texel.y < int(size.y)
            && texelFetch(body, texel, 0).r < dist - margin) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);  // behind the body: off screen
            return;
        }
    }
    vec2 p = mix(box.xy, box.zw, corner);
    // pixel (0, 0) top-left -> NDC, y up
    gl_Position = vec4(p.x / size.x * 2.0 - 1.0, 1.0 - p.y / size.y * 2.0, 0.0, 1.0);
    v_centre = centre;
    v_conic = conic;
    v_bgr = bgr;
    v_opacity = opacity;
}
"""

_FRAGMENT = """
#version 330 core
uniform vec2 size;
in vec2 v_centre;
in vec4 v_conic;
in vec3 v_bgr;
in float v_opacity;
out vec4 colour;
void main() {
    // the pixel's integer coordinates, rows counted from the top
    float rx = gl_FragCoord.x - 0.5;
    float ry = size.y - gl_FragCoord.y - 0.5;
    float dx = rx - v_centre.x;
    float dy = ry - v_centre.y;
    float power = -0.5 * (v_conic.z * dx * dx - 2.0 * v_conic.y * dx * dy + v_conic.x * dy * dy) / v_conic.w;
    float weight = v_opacity * exp(min(power, 0.0));
    if (weight <= 1.0 / 255.0) discard;
    colour = vec4(weight * v_bgr, weight);
}
"""


_BODY_VERTEX = """
#version 330 core
uniform mat4 view;                 // world -> camera (+z forward, +y down)
uniform vec4 camera;               // fx, fy, cx, cy
uniform vec2 size;
uniform vec2 zrange;               // near, far camera z, for the depth test
in vec3 position;
out vec3 v_camera;
void main() {
    vec4 c = view * vec4(position, 1.0);
    v_camera = c.xyz;
    // pixel coordinates as the splats use them (pixel centres at integers),
    // then the same pixel-edge -> NDC mapping; w = z for perspective-correct
    // interpolation of the camera-space position
    float ex = camera.x * c.x / c.z + camera.z + 0.5;
    float ey = camera.y * c.y / c.z + camera.w + 0.5;
    float depth = (c.z - zrange.x) / (zrange.y - zrange.x) * 2.0 - 1.0;
    gl_Position = vec4((ex / size.x * 2.0 - 1.0) * c.z, (1.0 - ey / size.y * 2.0) * c.z, depth * c.z, c.z);
}
"""

_BODY_FRAGMENT = """
#version 330 core
in vec3 v_camera;
out float distance;
void main() {
    distance = length(v_camera);
}
"""


_MASK_VERTEX = """
#version 330 core
uniform vec2 size;
in vec2 position;                  // pixel coordinates, pixel centres at integers
void main() {
    gl_Position = vec4((position.x + 0.5) / size.x * 2.0 - 1.0, 1.0 - (position.y + 0.5) / size.y * 2.0, 0.0, 1.0);
}
"""

_MASK_FRAGMENT = """
#version 330 core
out float inside;
void main() {
    inside = 1.0;
}
"""


class _GL:
    """One standalone context, programs and targets (one per process; the context belongs to its thread)."""

    def __init__(self):
        import moderngl

        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context(require=330)
        self.prog = self.ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.quad = self.ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], np.float32).tobytes())
        self.body_prog = self.ctx.program(vertex_shader=_BODY_VERTEX, fragment_shader=_BODY_FRAGMENT)
        self.mask_prog = self.ctx.program(vertex_shader=_MASK_VERTEX, fragment_shader=_MASK_FRAGMENT)
        self.mask_fbo = self.mask_shape = None
        self.fbo = None
        self.shape = None
        self.body_fbo = self.body_texture = None

    def target(self, height, width):
        if self.shape != (height, width):
            if self.fbo is not None:
                self.fbo.release()
                self.body_fbo.release()
                self.body_texture.release()
            colour = self.ctx.renderbuffer((width, height), components=4, dtype="f4")
            self.fbo = self.ctx.framebuffer(color_attachments=[colour])
            self.body_texture = self.ctx.texture((width, height), components=1, dtype="f4")
            self.body_texture.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
            self.body_fbo = self.ctx.framebuffer(color_attachments=[self.body_texture],
                                                 depth_attachment=self.ctx.depth_renderbuffer((width, height)))
            self.shape = (height, width)
        return self.fbo

    def body_distance(self, vertices, faces, view, intrinsics, height, width):
        """Draw the occluder's camera distance per pixel into `body_texture` (no surface: 1e9)."""
        vertices = np.ascontiguousarray(vertices, np.float32)
        camera_z = vertices @ np.asarray(view[2, :3], np.float32) + np.float32(view[2, 3])
        near = max(float(camera_z.min()) - 1.0, 1e-3)
        far = float(camera_z.max()) + 1.0
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(np.ascontiguousarray(faces, np.int32).tobytes())
        vao = self.ctx.vertex_array(self.body_prog, [(vbo, "3f", "position")], index_buffer=ibo, index_element_size=4)
        self.body_prog["view"].write(np.ascontiguousarray(np.asarray(view, np.float32).T).tobytes())
        self.body_prog["camera"].value = (float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2]))
        self.body_prog["size"].value = (float(width), float(height))
        self.body_prog["zrange"].value = (near, far)
        self.body_fbo.use()
        self.body_fbo.viewport = (0, 0, width, height)
        self.body_fbo.clear(1e9, 0.0, 0.0, 0.0, depth=1.0)
        self.ctx.disable(self.mgl.BLEND)
        self.ctx.enable(self.mgl.DEPTH_TEST)
        vao.render(self.mgl.TRIANGLES)
        self.ctx.disable(self.mgl.DEPTH_TEST)
        vao.release()
        vbo.release()
        ibo.release()


_STATE = None


def fill_triangles(pixels, faces, shape):
    """uint8 (H, W) mask, 1 inside the union of the 2D triangles `pixels[faces]`.

    Rasterised on the GPU (pixel in when its centre is), for the body
    silhouette; cv2.fillPoly was slower and left holes where triangles overlap."""
    global _STATE
    height, width = shape[:2]
    if _STATE is None:
        _STATE = _GL()
    gl = _STATE
    if gl.mask_shape != (height, width):
        if gl.mask_fbo is not None:
            gl.mask_fbo.release()
        gl.mask_fbo = gl.ctx.framebuffer(color_attachments=[gl.ctx.renderbuffer((width, height), components=1, dtype="f1")])
        gl.mask_shape = (height, width)
    vbo = gl.ctx.buffer(np.ascontiguousarray(pixels, np.float32).tobytes())
    ibo = gl.ctx.buffer(np.ascontiguousarray(faces, np.int32).tobytes())
    vao = gl.ctx.vertex_array(gl.mask_prog, [(vbo, "2f", "position")], index_buffer=ibo, index_element_size=4)
    gl.mask_prog["size"].value = (float(width), float(height))
    gl.mask_fbo.use()
    gl.mask_fbo.viewport = (0, 0, width, height)
    gl.mask_fbo.clear(0.0, 0.0, 0.0, 0.0)
    gl.ctx.disable(gl.mgl.BLEND)
    gl.ctx.disable(gl.mgl.DEPTH_TEST)
    vao.render(gl.mgl.TRIANGLES)
    raw = np.frombuffer(gl.mask_fbo.read(components=1, dtype="f1"), np.uint8).reshape(height, width)
    vao.release()
    vbo.release()
    ibo.release()
    return (raw[::-1] > 0).astype(np.uint8)


def render_splats_gl(splats, view, intrinsics, size, tile=3.0, bgra8=False, occluder=None, margin=0.03):
    """(BGR float image, alpha), as `splat_cpu.render_splats`.

    `occluder`: (vertices, faces) of a mesh that hides splats but is not
    drawn. A splat is skipped when the occluder's surface under its centre
    pixel is more than `margin` nearer the camera (the GPU form of
    `GarmentRenderer.occluded_by_body`).

    `bgra8`: return one (H, W, 4) uint8 array instead, rounded on the GPU
    (skips float64 frame copies)."""
    global _STATE
    height, width = size
    if bgra8:
        empty = np.zeros((height, width, 4), np.uint8)
    else:
        image = np.zeros((height, width, 3), np.float64)
        alpha = np.zeros((height, width), np.float64)
    if len(splats["xyz"]) == 0:
        return empty if bgra8 else (image, alpha)
    pixels, depth, valid, cov2d, determinant, colours, opacity, radius = _prepare(splats, view, intrinsics, tile)
    if splats.get("keep") is not None:
        # all Gaussians handed in, dropped ones masked (`GarmentRenderer.splats(full=True)`)
        valid &= np.asarray(splats["keep"], bool)
    # same per-splat pixel box as the CPU path
    x0 = np.maximum(np.floor(pixels[:, 0] - radius), 0)
    x1 = np.minimum(np.ceil(pixels[:, 0] + radius) + 1, width)
    y0 = np.maximum(np.floor(pixels[:, 1] - radius), 0)
    y1 = np.minimum(np.ceil(pixels[:, 1] + radius) + 1, height)
    valid &= (x1 > x0) & (y1 > y0)
    keep = np.flatnonzero(valid)
    if keep.size == 0:
        return empty if bgra8 else (image, alpha)
    keep = keep[far_to_near(depth[keep])]
    view64 = np.asarray(view, np.float64)
    instances = _instance_rows(len(keep))
    _instances(instances, keep, x0, y0, x1, y1, np.ascontiguousarray(pixels, np.float64), np.ascontiguousarray(cov2d, np.float64),
               np.ascontiguousarray(determinant, np.float64), np.ascontiguousarray(colours, np.float64),
               np.ascontiguousarray(np.asarray(opacity, np.float64).ravel()), np.ascontiguousarray(splats["xyz"], np.float64),
               np.ascontiguousarray(view64[:3, :3]), np.ascontiguousarray(view64[:3, 3]), occluder is not None)
    if _STATE is None:
        _STATE = _GL()
    gl = _STATE
    fbo = gl.target(height, width)
    use_body = occluder is not None and len(occluder[1]) > 0
    if use_body:
        gl.body_distance(occluder[0], occluder[1], view, intrinsics, height, width)
    gl.body_texture.use(location=0)  # bound even when unused, to avoid a GL warning
    buffer = gl.ctx.buffer(instances)
    vao = gl.ctx.vertex_array(
        gl.prog,
        [
            (gl.quad, "2f", "corner"),
            (buffer, "4f 2f 4f 3f 1f 1f 4x/i", "box", "centre", "conic", "bgr", "opacity", "dist"),
        ],
    )
    gl.prog["size"].value = (float(width), float(height))
    gl.prog["use_body"].value = int(use_body)
    gl.prog["margin"].value = float(margin)
    gl.prog["body"].value = 0
    fbo.use()
    fbo.viewport = (0, 0, width, height)
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    gl.ctx.enable(gl.mgl.BLEND)
    gl.ctx.blend_func = gl.mgl.ONE, gl.mgl.ONE_MINUS_SRC_ALPHA
    gl.ctx.disable(gl.mgl.DEPTH_TEST)
    vao.render(gl.mgl.TRIANGLE_STRIP, instances=len(keep))
    if bgra8:
        raw = np.frombuffer(fbo.read(components=4, dtype="f1"), np.uint8).reshape(height, width, 4)
        vao.release()
        buffer.release()
        return np.ascontiguousarray(raw[::-1])
    raw = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(height, width, 4)[::-1]
    vao.release()
    buffer.release()
    image[:] = raw[..., :3]
    alpha[:] = raw[..., 3]
    return image, alpha
