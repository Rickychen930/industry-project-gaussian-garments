"""`IntervalWorker` (tryon/garment/async_worker.py): the newest job's result,
at most one job per INTERVAL, a worker that stops on close."""

from __future__ import annotations

import threading
import time
import unittest

from tryon.garment.async_worker import IntervalWorker


class Doubler(IntervalWorker):
    INTERVAL = 0.0

    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.exited = threading.Event()
        super().__init__()

    def _make_job(self, value):
        return value

    def _compute(self, job):
        self.gate.wait(5.0)
        return job * 2

    def _closed_on_thread(self):
        self.exited.set()


def wait_for(condition, timeout=5.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.005)
    return False


class IntervalWorkerTest(unittest.TestCase):
    def test_a_submitted_job_becomes_the_latest_result(self):
        worker = Doubler()
        self.addCleanup(worker.close)
        self.assertIsNone(worker._newest())
        worker.submit(21)
        self.assertTrue(wait_for(lambda: worker._newest() == 42))
        self.assertEqual(worker.results, 1)

    def test_a_submit_while_busy_is_dropped(self):
        worker = Doubler()
        self.addCleanup(worker.close)
        worker.gate.clear()
        worker.submit(1)
        worker.submit(2)  # dropped: job 1 is running
        worker.gate.set()
        self.assertTrue(wait_for(lambda: worker._newest() == 2))
        time.sleep(0.05)
        self.assertEqual(worker._newest(), 2)
        self.assertEqual(worker.results, 1)

    def test_a_submit_inside_the_interval_is_dropped(self):
        worker = Doubler()
        self.addCleanup(worker.close)
        worker.INTERVAL = 60.0
        worker.submit(1)
        self.assertTrue(wait_for(lambda: worker._newest() == 2))
        self.assertTrue(wait_for(lambda: not worker._busy))
        worker.submit(5)
        time.sleep(0.05)
        self.assertEqual(worker._newest(), 2)

    def test_close_stops_the_thread_and_runs_the_exit_hook(self):
        worker = Doubler()
        worker.close()
        self.assertTrue(worker.exited.wait(5.0))


if __name__ == "__main__":
    unittest.main()
