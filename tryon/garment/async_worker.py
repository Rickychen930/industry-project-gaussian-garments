"""One worker thread that turns the newest job into a result, at most every
INTERVAL seconds: the scaffolding `shading_student.AsyncStudent` and
`appearance.AsyncAppearance` share (the live loop submits every frame, draws
with whatever finished last)."""

from __future__ import annotations

import threading
import time


class IntervalWorker:
    """Subclasses implement `_make_job(*args)` (copy what the thread needs;
    runs under the lock) and `_compute(job)` (runs on the worker thread; the
    value it returns is the new result). `_closed_on_thread()` runs once on
    the worker's way out."""

    INTERVAL = 0.0  # s between jobs: a submit sooner than this after the last accepted one is dropped

    def __init__(self):
        self._started = -1e9
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job = self._result = None
        self._busy = self._closed = False
        self.results = 0  # finished jobs
        threading.Thread(target=self._run, daemon=True).start()

    def close(self):
        """Stop the worker (a job already running finishes)."""
        self._closed = True
        self._wake.set()

    def _make_job(self, *args):
        raise NotImplementedError

    def _compute(self, job):
        raise NotImplementedError

    def _closed_on_thread(self):
        """Hook: the worker thread is about to exit."""

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._closed:
                self._closed_on_thread()
                return
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            try:
                result = self._compute(job)
                with self._lock:
                    self._result = result
                    self.results += 1
            finally:
                with self._lock:
                    self._busy = False

    def submit(self, *args):
        """Queue this frame unless a job is running or the last one started
        less than INTERVAL ago."""
        with self._lock:
            now = time.perf_counter()
            if self._busy or now - self._started < self.INTERVAL:
                return
            self._busy = True
            self._started = now
            self._job = self._make_job(*args)
        self._wake.set()

    def _newest(self):
        with self._lock:
            return self._result
