"""HTTP feature reads must not load the AIA download pipeline."""
import subprocess
import sys


def test_aia_feature_reads_do_not_import_downloader():
    result = subprocess.run(
        [sys.executable, '-c', (
            'import sys; '
            'import argus_clio.services.aia.features; '
            'assert "clio.dataloaders.aia" not in sys.modules; '
            'assert "argus_clio.services.aia.collection" not in sys.modules'
        )],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
