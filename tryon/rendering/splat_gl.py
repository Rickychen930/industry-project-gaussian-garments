"""The Gaussian compositing of `tryon/rendering/splat_cpu.py`'s `render_splats`, on the GPU.

`render_splats` spends most of a frame in `_composite_tiles`, a numba loop
over every (splat, pixel) pair on the CPU (~360 ms for a 93k-splat garment
at 480 x 640, and it grows with splat pixels). This draws the same thing
with OpenGL (moderngl, a standalone context - Metal-backed GL 4.1 on the
M1): the per-splat pre-pass is `splat._prepare`, shared unchanged, then
every splat is one instanced quad over exactly the pixel box the CPU path
uses, drawn farthest first, and its fragment evaluates the same Gaussian:

  power  = -0.5 (c dx^2 - 2 b dx dy + a dy^2) / det, clamped to <= 0
  weight = opacity * exp(power), discarded at <= 1/255
  out    = (weight * bgr, weight)  blended ONE, ONE_MINUS_SRC_ALPHA

Back-to-front "over" with premultiplied colour is the same recurrence as
the CPU's front-to-back transmittance sum, so the result matches it up to
float32 rounding (the CPU path is float64); the target is float32 RGBA so
nothing is quantised. Pixel (col, row) is sampled at its integer
coordinates, as on the CPU (dx = col - x_splat), by subtracting the 0.5
that gl_FragCoord carries. One difference: the CPU sorts by depth within
each tile, this sorts all splats once - the same order wherever two
splats overlap.

    image, alpha = render_splats_gl(splats, view, intrinsics, (height, width))
"""

from __future__ import annotations

import numpy as np
from numba import njit

from tryon.rendering.splat_cpu import _prepare


@njit(cache=True)
def _instances(order, x0, y0, x1, y1, pixels, cov2d, determinant, colours, opacity):
    """The per-instance attribute rows, in draw order, float32 - one pass
    instead of a numpy gather per column (22 -> ~1 ms at 121k splats).
    Serial: the loop is cheap, and a numba parallel region must not meet
    another thread's."""
    n = len(order)
    out = np.zeros((n, 16), np.float32)
    for i in range(n):
        k = order[i]
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = x0[k], y0[k], x1[k], y1[k]
        out[i, 4], out[i, 5] = pixels[k, 0], pixels[k, 1]
        out[i, 6], out[i, 7], out[i, 8] = cov2d[k, 0, 0], cov2d[k, 0, 1], cov2d[k, 1, 1]
        out[i, 9] = determinant[k]
        out[i, 10], out[i, 11], out[i, 12] = colours[k, 2], colours[k, 1], colours[k, 0]  # BGR
        out[i, 13] = opacity[k]
    return out

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
    """One standalone context, program and target, grown to the largest
    frame asked for (one per process: the context belongs to its thread)."""

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
        """Draw the occluder's camera distance per pixel into `body_texture`
        (nearest surface wins by the depth test; no surface: 1e9)."""
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
    """uint8 (H, W) mask, 1 inside the union of the 2D triangles
    `pixels[faces]` (pixel coordinates, centres at integers), rasterised on
    the GPU: a pixel is in when its centre is. For the body silhouette
    (21k triangles), where cv2.fillPoly took 15-98 ms and, drawing them all
    in one call, left holes where triangles overlap."""
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
    """(BGR float image, alpha), as `tryon.rendering.splat_cpu.render_splats`.

    `occluder`: (vertices, faces) of a mesh that hides splats but is not
    drawn (the wearer's body under a garment composited onto a photo). A
    splat is skipped when the occluder's surface under its centre pixel is
    more than `margin` nearer the camera than the splat - the ray cast of
    `GarmentRenderer.occluded_by_body`, done with a depth pass on the GPU.

    `bgra8`: return one (H, W, 4) uint8 array instead - the image and alpha
    rounded to 8 bits by the GPU on read-back, which is what a caller
    compositing onto a video frame wants (skips two float64 frame copies
    and the conversion, ~15 ms at 870 x 640)."""
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
        # every Gaussian was handed in, the dropped ones masked (see
        # `GarmentRenderer.splats(full=True)`)
        valid &= np.asarray(splats["keep"], bool)
    # the CPU path's per-splat pixel box, clipped to the image
    x0 = np.maximum(np.floor(pixels[:, 0] - radius), 0)
    x1 = np.minimum(np.ceil(pixels[:, 0] + radius) + 1, width)
    y0 = np.maximum(np.floor(pixels[:, 1] - radius), 0)
    y1 = np.minimum(np.ceil(pixels[:, 1] + radius) + 1, height)
    valid &= (x1 > x0) & (y1 > y0)
    keep = np.flatnonzero(valid)
    if keep.size == 0:
        return empty if bgra8 else (image, alpha)
    # farthest first; float64 depths practically never tie, so the unstable
    # sort gives the same order at half the cost (8.8 vs 16.6 ms, 121k)
    keep = keep[np.argsort(-depth[keep])]
    instances = _instances(keep, x0, y0, x1, y1, np.ascontiguousarray(pixels, np.float64), np.ascontiguousarray(cov2d, np.float64),
                           np.ascontiguousarray(determinant, np.float64), np.ascontiguousarray(colours, np.float64),
                           np.ascontiguousarray(np.asarray(opacity, np.float64).ravel()))
    if occluder is not None:
        centres = np.asarray(splats["xyz"], np.float64)[keep]
        instances[:, 14] = np.linalg.norm(centres @ np.asarray(view, np.float64)[:3, :3].T + np.asarray(view, np.float64)[:3, 3], axis=1)
    if _STATE is None:
        _STATE = _GL()
    gl = _STATE
    fbo = gl.target(height, width)
    use_body = occluder is not None and len(occluder[1]) > 0
    if use_body:
        gl.body_distance(occluder[0], occluder[1], view, intrinsics, height, width)
    gl.body_texture.use(location=0)  # bound either way: an empty sampler unit logs a GL warning
    buffer = gl.ctx.buffer(instances.tobytes())
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
