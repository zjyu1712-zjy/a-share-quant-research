import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class V2FactorCliTests(unittest.TestCase):
    def test_help_lists_v2_factor_commands(self):
        result = subprocess.run(
            ["./run_mac.sh", "help"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("v2factordata", result.stdout)
        self.assertIn("v2factors", result.stdout)

    def test_screening_cli_exits_two_when_data_audit_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            (snapshots / "2020-01-31.csv").write_text(
                "code\n600000\n", encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.v2_factor_screening",
                    "--snapshot-dir",
                    str(snapshots),
                    "--cache-root",
                    str(root / "cache"),
                    "--output-dir",
                    str(root / "output"),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("data_audit_failed", result.stdout)

    def test_missing_universe_fails_closed_with_report_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run(
                [sys.executable, "-m", "src.v2_factor_screening",
                 "--snapshot-dir", str(root / "missing"),
                 "--cache-root", str(root / "cache"),
                 "--output-dir", str(root / "output")],
                cwd=PROJECT_ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            summary = json.loads(
                (root / "output" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "data_audit_failed")
            self.assertFalse(summary["deployment_authorized"])
            self.assertTrue((root / "output" / "report.html").is_file())
            self.assertFalse((root / "output" / "factor_values.csv").exists())

    def test_factor_data_cli_rejects_invalid_universe_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_dir = root / "snapshots"
            snapshot_dir.mkdir()
            (snapshot_dir / "2020-01-31.csv").write_text(
                "code\n600000\n", encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.v2_factor_data",
                    "--snapshot-dir",
                    str(snapshot_dir),
                    "--cache-root",
                    str(root / "cache"),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("股票池并集1只", result.stdout)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
