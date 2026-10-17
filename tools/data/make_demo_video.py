"""Build the demo video: camera frames next to the live try-on, with title, results and end cards.

Record your own with the launcher, then point this tool at the two files it saves:

    ./live-tryon bench 00190_upper 200          # stand in view; saves bench-<time>-input.mp4 and bench-<time>.mp4
    python -m tools.data.make_demo_video --camera ~/tryon-data/captures/bench-<time>-input.mp4 \\
        --tryon ~/tryon-data/captures/bench-<time>.mp4 --out docs/demo/demo.mp4

The camera video and the try-on video hold the same frames, so they are shown frame for frame.
ffmpeg and Pillow are needed. The text on the cards is in `CARDS` below; edit it to match your own run.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
BACKGROUND, INK, MUTE, ACCENT = (20, 24, 29), (238, 241, 244), (150, 162, 174), (224, 120, 70)
FONT = "/System/Library/Fonts/Helvetica.ttc"
ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-crf", "20", "-an"]

CARDS = {
    "title": ([("Live 3D Virtual Try-On from a Webcam", 76, True, INK),
               ("Captured Gaussian-Splat Garments on an Apple M1", 46, False, ACCENT),
               ("", 20, False, INK),
               ("32040 Industry Project, UTS, Spring 2026", 36, False, MUTE),
               ("Jayson M. Hendra, Michael P. Budiman, Ricky", 36, False, MUTE)],
              "Supervisor: Professor Stuart Perry", 4),
    "results": ([("Results", 70, True, INK), ("", 20, False, INK),
                 ("Body error 24.6 mm on 16 annotated views (42.5 mm on 4,208 views)", 42, False, INK),
                 ("Garment overlap (IoU) 0.730; authors' four-camera registration 0.827", 42, False, INK),
                 ("0.1 to 0.2 s of work per frame; about 25 fps in smooth mode, 7.3 to 7.5 fps default",
                  42, False, INK),
                 ("Garment present in 98.6% of 13,178 webcam frames", 42, False, INK),
                 ("Open: IoU target 0.80, erased arms, 40 dB shading goal", 42, False, ACCENT)],
                "All numbers are measured on one MacBook with an Apple M1 and no CUDA GPU", 9),
    "end": ([("Code, tests and measurements", 60, True, INK),
             ("github.com/Rickychen930/industry-project-gaussian-garments", 44, False, ACCENT),
             ("", 20, False, INK),
             ("Wearer: Ricky, a team member, who gave consent", 34, False, MUTE),
             ("Garments: Gaussian Garments (Rong et al., 2025)", 34, False, MUTE)], None, 4),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def _card(path: Path, lines, footer) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    y = 250
    for text, size, bold, color in lines:
        font = _font(size, bold)
        draw.text(((WIDTH - draw.textlength(text, font=font)) / 2, y), text, font=font, fill=color)
        y += int(size * 1.55)
    if footer:
        font = _font(30)
        draw.text(((WIDTH - draw.textlength(footer, font=font)) / 2, HEIGHT - 120), footer, font=font, fill=MUTE)
    image.save(path)


def _labels(path: Path, caption: str) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((40, 205), "Camera", font=_font(40, True), fill=INK + (255,))
    draw.text((1000, 205), "Live try-on", font=_font(40, True), fill=INK + (255,))
    draw.text((40, 880), caption, font=_font(36), fill=MUTE + (255,))
    image.save(path)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def _card_clip(work: Path, name: str, fade: bool) -> str:
    lines, footer, length = CARDS[name]
    _card(work / f"{name}.png", lines, footer)
    effect = "scale=1920:1080,format=yuv420p" + (",fade=t=in:st=0:d=0.5" if fade else "")
    _ffmpeg("-loop", "1", "-t", str(length), "-i", str(work / f"{name}.png"), "-vf", effect, *ENCODE,
            str(work / f"{name}.mp4"))
    return f"{name}.mp4"


def build(camera: Path, tryon: Path, out: Path, seconds: int, caption: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        parts = [_card_clip(work, "title", fade=True)]
        _labels(work / "labels.png", caption)
        frames = seconds * 10
        graph = (f"[0:v]select='lt(n,{frames})',setpts=N/(10*TB),scale=960:540[a];"
                 f"[1:v]select='lt(n,{frames})',setpts=N/(10*TB),scale=960:540[b];"
                 "[a][b]hstack=inputs=2,fps=30[v];color=c=0x14181d:s=1920x1080:r=30[bg];"
                 "[bg][v]overlay=0:270[o];[o][2:v]overlay=0:0:shortest=1,format=yuv420p")
        _ffmpeg("-i", str(camera), "-i", str(tryon), "-loop", "1", "-i", str(work / "labels.png"),
                "-filter_complex", graph, "-t", str(seconds), *ENCODE, str(work / "clip.mp4"))
        parts.append("clip.mp4")
        parts += [_card_clip(work, "results", fade=False), _card_clip(work, "end", fade=False)]
        (work / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
        out.parent.mkdir(parents=True, exist_ok=True)
        _ffmpeg("-f", "concat", "-safe", "0", "-i", str(work / "list.txt"), "-vf", "scale=1280:720",
                "-c:v", "libx264", "-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=Path, required=True, help="the saved camera frames, bench-<time>-input.mp4")
    parser.add_argument("--tryon", type=Path, required=True, help="the saved try-on output, bench-<time>.mp4")
    parser.add_argument("--out", type=Path, default=Path("docs/demo/demo.mp4"))
    parser.add_argument("--seconds", type=int, default=20, help="length of the side-by-side part")
    parser.add_argument("--caption", default="The garment follows the wearer. The HUD shows frame rate and reprojection error.")
    args = parser.parse_args()
    build(args.camera.expanduser(), args.tryon.expanduser(), args.out, args.seconds, args.caption)
    print(args.out)


if __name__ == "__main__":
    main()
