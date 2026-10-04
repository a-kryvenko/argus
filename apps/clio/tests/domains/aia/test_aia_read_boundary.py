"""HTTP original-file reads must not load the SDO download pipeline."""
import subprocess
import sys


def test_aia_original_reads_do_not_import_downloader():
    result = subprocess.run(
        [sys.executable, '-c', (
            'import sys; '
            'import clio.routers.observation_files; '
            'assert "clio.providers.sdo_images" not in sys.modules; '
            'assert "clio.commands.sdo_images" not in sys.modules'
        )],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
