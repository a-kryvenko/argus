"""The SDO provider is independent of ingestion."""
import subprocess
import sys


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
