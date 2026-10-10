"""Every Python file of the package, the tools and the launchers at least compiles.

The evaluation tools are not imported by the other tests; an edit that broke the syntax of one (10 Oct 2026:
`tools/evaluate/vs_registration.py`, after removing a line by string replacement) shipped with a green suite.
"""

import py_compile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class SourcesCompileTest(unittest.TestCase):
    def test_every_python_file_compiles(self):
        failures = []
        with tempfile.TemporaryDirectory() as scratch:
            for folder in ("tryon", "tools", "tests"):
                for path in sorted((ROOT / folder).rglob("*.py")):
                    try:
                        py_compile.compile(str(path), cfile=str(Path(scratch) / "x.pyc"), doraise=True)
                    except py_compile.PyCompileError as error:
                        failures.append(f"{path.relative_to(ROOT)}: {error.msg.strip().splitlines()[-1]}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
