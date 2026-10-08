"""setup.sh step 4: scan-pose assets are installed into checkpoints that have none."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _function():
    text = (ROOT / "setup.sh").read_text()
    return re.search(r"^install_scan_poses\(\) \{.*?^\}", text, re.S | re.M).group(0)


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class InstallScanPosesTest(unittest.TestCase):
    def test_assets_are_installed_where_missing_and_never_over_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            assets, garments = tmp / "assets", tmp / "garments"
            assets.mkdir()
            for name in ("a", "b"):
                (garments / name / "stage1").mkdir(parents=True)
                np.savez(assets / f"{name}.npz", betas=np.zeros(1), refined=np.asarray(1))
            np.savez(garments / "b" / "stage1" / "scan_pose.npz", betas=np.ones(1))
            script = f'ROOT={ROOT}\n{_function()}\ninstall_scan_poses "{garments}" "{assets}"'
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
            self.assertIn("installed a", result.stdout)
            self.assertNotIn("installed b", result.stdout)
            self.assertEqual(int(np.load(garments / "a" / "stage1" / "scan_pose.npz")["refined"]), 1)
            self.assertEqual(float(np.load(garments / "b" / "stage1" / "scan_pose.npz")["betas"][0]), 1.0)


if __name__ == "__main__":
    unittest.main()
