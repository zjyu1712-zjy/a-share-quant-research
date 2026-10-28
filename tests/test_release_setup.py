"""First-run checks for a checkout without private data or credentials."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@unittest.skipIf(sys.platform == "win32", "macOS/Linux launcher")
class FirstRunTests(unittest.TestCase):
    def test_doctor_succeeds_without_news_database_and_does_not_create_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(ROOT / "run_mac.sh", root / "run_mac.sh")
            (root / "src").mkdir()
            shutil.copy2(ROOT / "src" / "config.py", root / "src" / "config.py")
            (root / ".venv").symlink_to(sys.prefix, target_is_directory=True)
            result = subprocess.run(
                ["bash", str(root / "run_mac.sh"), "doctor"],
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((root / "data" / "financial_news.db").exists())


if __name__ == "__main__":
    unittest.main()
