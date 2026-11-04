"""HTTP original-file reads must not load the SDO download pipeline."""
import subprocess
import sys


def test_aia_original_reads_do_not_import_downloader():
    result = subprocess.run(
        [sys.executable, '-c', (
            'import sys; '
            'import clio.routers.observation_files; '
            'assert "clio.providers.sdo" not in sys.modules; assert "clio.ingestion.extras.sdo" not in sys.modules'
        )],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr



def test_sdo_provider_does_not_import_ingestion():
    result = subprocess.run(
        [sys.executable, '-c', (
            'import sys; '
            'import clio.providers.sdo; '
            'assert not any(name.startswith("clio.ingestion") for name in sys.modules)'
        )],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
