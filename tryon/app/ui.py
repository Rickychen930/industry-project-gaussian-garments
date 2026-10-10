"""The on-screen controls of the live window: pick a garment, take a picture, quit.

An `Overlay` is drawn over a copy of every frame the window shows (never over
the frames written to a video) and turns mouse clicks and keys into actions:

    ("select", i)   wear garment i of the wardrobe
    ("exit",)       close the window

Everything else (paging the garment bar, hiding the controls, saving a
picture) it handles itself. The window loops call `render` then `poll`.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from tryon.paths import CAPTURES

WINDOW = "Gaussian Garments try-on"
REFERENCE_WIDTH = 960  # the frame width the sizes below are written for
LOAD_BANNER_SECONDS = 4.0
TOAST_SECONDS = 2.0
CHIP_WIDTH, CHIP_GAP, BAR_HEIGHT, MARGIN, BUTTON_HEIGHT = 190, 8, 84, 10, 36

# garment -> (label, kind); kind: closed = cleanest result, open front = jacket-like, bottom = trousers / skirt
GARMENTS = {
    "00122_outer": ("Plaid shirt", "open front"),
    "00134_upper": ("Brown shirt", "short sleeve"),
    "00136_outer": ("Denim jacket", "open front"),
    "00148_outer": ("Beige blazer", "open front"),
    "00148_upper": ("Black dress", "short sleeve"),
    "00152_lower": ("Denim shorts", "bottom"),
    "00152_outer": ("Red plaid shirt", "open front"),
    "00170_lower": ("Black mini skirt", "bottom"),
    "00170_upper": ("Vest + shirt", "closed"),
    "00176_lower": ("Blue jeans", "bottom"),
    "00176_outer": ("White jacket", "open front"),
    "00176_upper": ("White T-shirt", "closed"),
    "00190_upper": ("Striped sweater", "closed"),
    "10004_outer": ("Green fur jacket", "open front"),
    "10006_outer": ("Pink fur jacket", "open front"),
}

WHITE, BLACK, ACCENT, DANGER, GREY = (255, 255, 255), (20, 20, 20), (150, 200, 0), (60, 60, 210), (90, 90, 90)
KEY_ESCAPE = 27
NO_KEY = (-1, 255)


def describe(name):
    """(label, kind) of garment `name`; unknown checkpoints show their folder name."""
    return GARMENTS.get(name, (name, ""))


def scale_of(width):
    return max(0.6, min(2.0, width / REFERENCE_WIDTH))


def layout(width, height, count, page, hidden):
    """Where every control sits in a frame of `width` x `height`, as
    {name: (x0, y0, x1, y1)} with the garment chips under "chip:<index>"
    and the bar behind them under "bar". Also returns how many chips fit
    on a page."""
    s = scale_of(width)
    margin, gap, button = int(MARGIN * s), int(CHIP_GAP * s), int(BUTTON_HEIGHT * s)
    rects = {}
    right = width - margin
    if hidden:
        rects["show"] = (right - int(64 * s), margin, right, margin + button)
        return rects, 0
    rects["exit"] = (right - int(88 * s), margin, right, margin + button)
    rects["snapshot"] = (rects["exit"][0] - gap - int(118 * s), margin, rects["exit"][0] - gap, margin + button)
    rects["hide"] = (rects["snapshot"][0] - gap - int(100 * s), margin, rects["snapshot"][0] - gap, margin + button)
    if count == 0:
        return rects, 0
    bar_top = height - int(BAR_HEIGHT * s)
    top, bottom = bar_top + int(12 * s), height - int(12 * s)
    arrow = int(40 * s)
    first, last = margin + arrow + gap, width - margin - arrow - gap
    chip = int(CHIP_WIDTH * s)
    per_page = max(1, (last - first + gap) // (chip + gap))
    rects["bar"] = (0, bar_top, width, height)
    rects["prev"] = (margin, top, margin + arrow, bottom)
    rects["next"] = (width - margin - arrow, top, width - margin, bottom)
    pages = -(-count // per_page)
    start = (page % pages) * per_page
    used = min(per_page, count - start) * (chip + gap) - gap  # the page's chips, centred between the arrows
    x = first + ((last - first) - used) // 2
    for index in range(start, min(start + per_page, count)):
        rects[f"chip:{index}"] = (x, top, x + chip, bottom)
        x += chip + gap
    return rects, per_page


def _inside(rect, x, y):
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


class Overlay:
    """The controls of one window. `names`: the garments (checkpoint folder
    names) the bar offers, `current`: the one worn now."""

    def __init__(self, names=(), current=0, snapshot_dir=CAPTURES):
        self.names, self.current = list(names), current
        self.snapshot_dir = Path(snapshot_dir)
        self.page, self.hidden, self.closed = 0, False, False
        self._click, self._hover, self._clean, self._shape = None, None, None, None
        self._banner, self._toast = ("", 0.0), ("", 0.0)  # (text, until)
        self._per_page = 1
        self._seen, self._opened, self._follow = False, False, True

    @classmethod
    def for_garment(cls, garment):
        """The overlay for a run wearing checkpoint `garment` (None: just the buttons)."""
        if garment is None:
            return cls()
        from tryon.app.pipeline import wardrobe_of

        wardrobe, current = wardrobe_of(garment)
        return cls([p.name for p in wardrobe], current)

    # -- mouse ------------------------------------------------------------

    def attach(self):
        """Create the window and listen to the mouse in it."""
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._mouse)

    def _mouse(self, event, x, y, flags, param):
        self._hover = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self._click = (x, y)

    def _hit(self, point):
        """The control under `point` (x, y), or None."""
        if self._shape is None or point is None:
            return None
        rects, _ = layout(*self._shape, len(self.names), self.page, self.hidden)
        return next((name for name, rect in rects.items() if name != "bar" and _inside(rect, *point)), None)

    # -- actions ----------------------------------------------------------

    def _wear(self, index):
        if not self.names or index == self.current:
            return None
        self.current, self._follow = index % len(self.names), True
        label = describe(self.names[self.current])[0]
        self._banner = (f"Loading {label} ...", time.monotonic() + LOAD_BANNER_SECONDS)
        return ("select", self.current)

    def _window_gone(self):
        """True once the window that was open has been closed with its own
        button (a backend that never reports it visible is not taken for closed)."""
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
            self._opened = True
            return False
        return self._opened

    def _key_control(self, key):
        """The control a key stands for: q / Esc exit, s snapshot, h hide, 1-9 a garment."""
        if ord("1") <= key <= ord("9") and key - ord("1") < len(self.names):
            return f"chip:{key - ord('1')}"
        return {ord("q"): "exit", KEY_ESCAPE: "exit", ord("s"): "snapshot", ord("h"): "hide"}.get(key)

    def poll(self, key=-1):
        """The action of the click or key since the last call, or None.
        `key`: what cv2.waitKey returned (& 0xFF; 255 / -1: none)."""
        if not self._seen:
            return None
        if self._window_gone():
            self.closed = True
            return ("exit",)
        click, self._click = self._click, None
        control = self._hit(click) or (None if key in NO_KEY else self._key_control(key))
        if key in (ord("n"), ord("p")) and self.names:
            return self._wear((self.current + (1 if key == ord("n") else -1)) % len(self.names))
        if control == "exit":
            return ("exit",)
        if control == "snapshot":
            self._save_picture()
        elif control in ("hide", "show"):
            self.hidden = not self.hidden
        elif control in ("prev", "next"):
            pages = -(-len(self.names) // self._per_page)
            self.page = (self.page + (1 if control == "next" else -1)) % pages
        elif control is not None and control.startswith("chip:"):
            return self._wear(int(control.split(":")[1]))
        return None

    def _save_picture(self):
        if self._clean is None:
            return
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"tryon-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        ok = cv2.imwrite(str(path), self._clean)
        self._toast = (f"Saved {path.name}" if ok else "Could not save the picture", time.monotonic() + TOAST_SECONDS)
        print(f"picture {path}" if ok else "could not save the picture", flush=True)

    # -- drawing ----------------------------------------------------------

    def render(self, frame):
        """A copy of `frame` with the controls drawn on it."""
        self._clean, self._seen = frame, True
        height, width = frame.shape[:2]
        self._shape = (width, height)
        out = frame.copy()
        s = scale_of(width)
        rects, per_page = layout(width, height, len(self.names), self.page, self.hidden)
        self._per_page = per_page or self._per_page
        if self._follow and per_page:  # open on the page that holds the garment worn
            self.page, self._follow = self.current // per_page, False
            rects, _ = layout(width, height, len(self.names), self.page, self.hidden)
        if "bar" in rects:
            x0, y0, x1, y1 = rects["bar"]
            out[y0:y1, x0:x1] = (out[y0:y1, x0:x1] * 0.35).astype(np.uint8)
        hover = self._hit(self._hover)
        for name, rect in rects.items():
            if name == "bar":
                continue
            lit = name == hover
            if name.startswith("chip:"):
                self._chip(out, rect, int(name.split(":")[1]), lit, s)
            elif name in ("prev", "next"):
                self._arrow(out, rect, name == "next", lit)
            else:
                label, fill = {"exit": ("Exit", DANGER), "snapshot": ("Snapshot", GREY),
                               "hide": ("Hide UI", GREY), "show": ("UI", GREY)}[name]
                self._button(out, rect, label, fill, lit, s)
        if not self.hidden:
            self._notices(out, s)
        return out

    def banner(self, frame, text):
        """`frame` with the controls and `text` in its middle (for a moment the loop is about to block)."""
        out = self.render(frame)
        self._text(out, text, (out.shape[1] // 2, out.shape[0] // 2), 0.8 * scale_of(out.shape[1]), centre=True, pill=True)
        return out

    def _notices(self, out, s):
        now = time.monotonic()
        height, width = out.shape[:2]
        for (text, until), y in ((self._banner, height // 2), (self._toast, height - int(BAR_HEIGHT * s) - int(24 * s))):
            if text and now < until:
                self._text(out, text, (width // 2, y), 0.8 * s, centre=True, pill=True)

    @staticmethod
    def _text(out, text, origin, scale, centre=False, colour=WHITE, pill=False):
        thickness = max(1, round(scale * 2))
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = (origin[0] - w // 2, origin[1] + h // 2) if centre else origin
        if pill:
            pad = int(10 * scale)
            cv2.rectangle(out, (x - pad, y - h - pad), (x + w + pad, y + base + pad), BLACK, -1)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)

    def _button(self, out, rect, label, fill, lit, s):
        x0, y0, x1, y1 = rect
        cv2.rectangle(out, (x0, y0), (x1, y1), tuple(min(255, c + 50) for c in fill) if lit else fill, -1)
        self._text(out, label, ((x0 + x1) // 2, (y0 + y1) // 2), 0.55 * s, centre=True)

    @staticmethod
    def _arrow(out, rect, right, lit):
        x0, y0, x1, y1 = rect
        cx, cy, r = (x0 + x1) // 2, (y0 + y1) // 2, (x1 - x0) // 4
        sign = 1 if right else -1
        cv2.rectangle(out, (x0, y0), (x1, y1), (110, 110, 110) if lit else (70, 70, 70), -1)
        triangle = np.array([[cx + sign * r, cy], [cx - sign * r, cy - r], [cx - sign * r, cy + r]], np.int32)
        cv2.fillPoly(out, [triangle], WHITE)

    def _chip(self, out, rect, index, lit, s):
        x0, y0, x1, y1 = rect
        worn = index == self.current
        cv2.rectangle(out, (x0, y0), (x1, y1), ACCENT if worn else ((110, 110, 110) if lit else (70, 70, 70)), -1)
        label, kind = describe(self.names[index])
        ink, pad = (BLACK if worn else WHITE), int(10 * s)
        title = self._fit(label, x1 - x0 - 2 * pad - int(22 * s), 0.6 * s)
        self._text(out, f"{index + 1}  {title}", (x0 + pad, y0 + int(26 * s)), 0.6 * s, colour=ink)
        self._text(out, kind, (x0 + pad, y0 + int(48 * s)), 0.45 * s, colour=ink)

    @staticmethod
    def _fit(text, room, scale):
        """`text` cut to fit `room` pixels."""
        thickness = max(1, round(scale * 2))
        while len(text) > 3 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > room:
            text = text[:-2].rstrip() + "."
        return text
