"""Which camera the live try-on opens: the laptop's own, never the phone's.

macOS offers an iPhone near the Mac as a camera of its own (Continuity Camera), and it can take the
place of index 0. `--source` therefore defaults to "builtin": the index of the laptop's camera, found by
name. A number, or a video file, is used as given.

    list_cameras()       [(index, name)] of the video devices, in the order OpenCV numbers them
    builtin_index()      the index of the laptop's camera (0 when it cannot be told)
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
    """[(index, name)] of the video devices in `ffmpeg -f avfoundation -list_devices true -i ""`'s output
    (the screens it lists as 'Capture screen N' and the audio devices are left out)."""
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
    """The index of the laptop's camera among `cameras` [(index, name)]: the first whose name is a built-in
    one, else the first that is not a phone, else the first, else 0."""
    for index, name in cameras:
        if any(word in name.lower() for word in _BUILTIN) and not any(word in name.lower() for word in _PHONE):
            return index
    for index, name in cameras:
        if not any(word in name.lower() for word in _PHONE):
            return index
    return cameras[0][0] if cameras else 0


def builtin_index():
    return choose_builtin(list_cameras())


def resolve_source(source):
    """`source` with the names of `BUILTIN_ALIASES` replaced by the laptop camera's index (a string, as `--source`
    is); a camera number or a file path comes back as it was."""
    if str(source).lower() in BUILTIN_ALIASES:
        cameras = list_cameras()
        index = choose_builtin(cameras)
        print(f"camera {index}: {dict(cameras).get(index, 'default')}", flush=True)
        return str(index)
    return str(source)
