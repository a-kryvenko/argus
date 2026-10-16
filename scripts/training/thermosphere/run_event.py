"""Run the corresponding command from the private research checkout."""
from importlib import import_module
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    directory = import_module("notebooks.lib.development").command_path('training/thermosphere')
    sys.path.insert(0, directory)
    runpy.run_path(str(Path(directory) / 'run_event.py'), run_name="__main__")
