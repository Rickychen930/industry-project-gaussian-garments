"""Command line of the live try-on: `python -m tryon.app [flags]`.

The launchers (`./live-tryon run`, `./live-tryon`) call this with the
default flag set; `--help` lists every flag, grouped by what it changes.
"""

from __future__ import annotations

import argparse

from tryon.app.pipeline import run
from tryon.app.smooth_display import smooth_display
from tryon.garment.renderer import GarmentRenderer


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m tryon.app", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    io = parser.add_argument_group("input and output")
    io.add_argument("--source", default="0", help="camera index or video file")
    io.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0: never)")
    io.add_argument("--max-width", type=int, default=960,
                    help="process frames at most this wide (0: as captured); 720 is ~17 ms a frame faster, same accuracy, smaller picture")
    io.add_argument("--headless", action="store_true", help="no window")
    io.add_argument("--output", help="write the last frame here (synchronous display)")
    io.add_argument("--output-video", help="write every shown frame (with the HUD) to this .mp4")
    io.add_argument("--profile", action="store_true", help="print the median time of each stage at the end")

    garment = parser.add_argument_group("garment")
    garment.add_argument("--garment", help="garment checkpoint directory; omit to draw the fitted body only")
    garment.add_argument("--garment-subsample", type=float, default=None,
                         help="fraction of the garment's Gaussians to draw, e.g. 0.5 (fewer splats, less detail); omit for all")
    garment.add_argument("--rigid", action="store_true", help="no cloth motion: the garment follows the body by skinning only")
    garment.add_argument("--appearance", action="store_true",
                         help="pose-dependent shading from the checkpoint's appearance network (with --student-shading: its "
                              "distilled student) instead of the static colours")
    garment.add_argument("--student-shading", action="store_true",
                         help="with --appearance: the garment's distilled shading student when it has one")
    garment.add_argument("--async-appearance", action="store_true",
                         help="with --appearance: shade on a worker thread, each frame uses the newest shading (live speed)")
    garment.add_argument("--gpu-render", action="store_true", help="composite the splats with OpenGL (tryon/rendering/splat_gl.py)")

    body = parser.add_argument_group("body")
    body.add_argument("--body-regressor", default=None,
                      help="the body from an image regressor: nlf (NLF-S, the default with --garment when installed) or a "
                           "Multi-HMR model name (e.g. multiHMR_672_S); 'none' fits it to MediaPipe's keypoints alone")
    body.add_argument("--orientation-regressor",
                      help="models/pose_regressor.pt: the facing classifier's seeds for a keypoint cold fit and its "
                           "disagreement check while tracking (tools/train/orientation.py)")

    removal = parser.add_argument_group("removing the wearer's own clothing")
    removal.add_argument("--remove-clothing", action="store_true",
                         help="erase the wearer's clothing before the garment is drawn")
    removal.add_argument("--parser", choices=["schp", "fashn"], default=None,
                         help="clothing parser: schp (the default, ~0.1 s on CoreML) or fashn (more accurate, "
                              "~0.8 s: for --output-video runs)")
    removal.add_argument("--fill", choices=["skin", "lama"], default=None,
                         help="fill of the erased clothing: skin (Telea toward the skin colour, the default) or lama "
                              "(Big-LaMa, ~4 s a frame: for --output-video runs)")
    removal.add_argument("--no-background-plate", action="store_true",
                         help="do not learn the room behind the wearer (tryon/removal/background.py)")
    removal.add_argument("--harmonize", action="store_true",
                         help="harmonise the garment's colour to the scene (tryon/app/harmonize.py; ~60 ms a frame)")

    display = parser.add_argument_group("display")
    display.add_argument("--smooth-display", action="store_true",
                         help="show every camera frame; the pipeline runs behind it and its newest garment is moved with the torso")
    display.add_argument("--delayed-sync", action="store_true",
                         help="with --smooth-display: show the video ~1 pipeline period late, the garment from the nearest pipeline frame")
    display.add_argument("--dump-alpha", default=None,
                         help="with --smooth-display: save the garment alpha of every shown frame (npz), for evaluation")
    return parser


def apply_renderer_switches(args):
    """The renderer's switches are class attributes, so the smooth
    display's pipeline process gets them too."""
    GarmentRenderer.GPU_RASTER = GarmentRenderer.GPU_RASTER or args.gpu_render
    GarmentRenderer.APPEARANCE_ASYNC = args.async_appearance
    GarmentRenderer.STUDENT_SHADING = args.student_shading
    GarmentRenderer.DYNAMICS = not args.rigid


def pipeline_options(args):
    """The pipeline's own options, the same in both display modes."""
    return dict(
        garment=args.garment, garment_subsample=args.garment_subsample, orientation_regressor=args.orientation_regressor,
        remove_clothing=args.remove_clothing, parser=args.parser, fill=args.fill, appearance=args.appearance,
        background_plate=not args.no_background_plate, harmonize=args.harmonize, body_regressor=args.body_regressor,
        profile=args.profile)


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_renderer_switches(args)
    if args.smooth_display:
        return smooth_display(args.source, max_width=args.max_width, headless=args.headless, output_video=args.output_video,
                              max_frames=args.max_frames, dump_alpha=args.dump_alpha, delayed=args.delayed_sync, **pipeline_options(args))
    return run(args.source, max_frames=args.max_frames, headless=args.headless, output=args.output, output_video=args.output_video,
               max_width=args.max_width, **pipeline_options(args))
