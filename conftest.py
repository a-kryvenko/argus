"""Make shared integration helpers available to repository test suites."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "tests/support"))
