"""Which camera the live try-on opens: the laptop's own, never the phone's.

macOS can offer a nearby iPhone as a camera (Continuity Camera) in place of index 0, so
`--source` defaults to "builtin": the laptop camera's index, found by name. A number or
a video file is used as given.

    list_cameras()       [(index, name)] of the video devices, in the order OpenCV numbers them
    resolve_source(s)    "builtin" / "laptop" / "auto" -> the index as a string; anything else unchanged
"""

from __future__ import annotations

import re
import shutil
import subprocess

BUILTIN_ALIASES = ("builtin", "laptop", "auto")
# names of the laptop's own camera, and of the ones that must not be picked by default
_BUILTIN = ("facetime", "built-in", "builtin", "isight", "macbook", "imac")
_PHONE = ("iphone", "ipad", "continuity", "desk view")
_DEVICE_LINE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


def parse_avfoundation_listing(text):
    """[(index, name)] of the video devices in `ffmpeg -f avfoundation -list_devices true -i ""`
    (screens and audio devices left out)."""
    cameras, in_video = [], False
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        match = _DEVICE_LINE.search(line)
        if in_video and match and not match.group(2).lower().startswith("capture screen"):
            cameras.append((int(match.group(1)), match.group(2)))
    return cameras


def list_cameras():
    """The camera list from ffmpeg when it is installed; [] when it cannot be told."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return []
    try:
        run = subprocess.run([ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_avfoundation_listing(run.stderr + run.stdout)


def choose_builtin(cameras):
    """The index of the laptop's camera among `cameras`: the first built-in name, else the
    first that is not a phone, else the first, else 0."""
    for index, name in cameras:
        if any(word in name.lower() for word in _BUILTIN) and not any(word in name.lower() for word in _PHONE):
            return index
    for index, name in cameras:
        if not any(word in name.lower() for word in _PHONE):
            return index
    return cameras[0][0] if cameras else 0


def resolve_source(source):
    """`source` with `BUILTIN_ALIASES` replaced by the laptop camera's index (a string);
    a camera number or file path comes back as it was."""
    if str(source).lower() in BUILTIN_ALIASES:
        cameras = list_cameras()
        index = choose_builtin(cameras)
        print(f"camera {index}: {dict(cameras).get(index, 'default')}", flush=True)
        return str(index)
    return str(source)
