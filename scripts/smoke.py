"""Run the small CPU smoke check without installing a test runner."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    checks = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tests/test_smoke.py")
    )
    checks["test_smoke"]()
    print("CPU smoke passed")
