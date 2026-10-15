"""Infrastructure launcher; audit implementation lives in private intelligence-core."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'packages/intelligence-core/tools/drag_physics_audit.py'), run_name='__main__')
