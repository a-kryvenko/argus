import pytest

@pytest.mark.parametrize('quality, archive, nrt', [
    (0, True, True), (0x40000000, False, True),
    (-1, False, False), (1, False, False),
    (0x40002000, False, False), (0x80000000, False, False),
])
def test_quality_only_allows_nrt_mode_bit(quality, archive, nrt):
    from argus_prophet.services.aia.extraction import valid_quality
    assert valid_quality(quality) is archive
    assert valid_quality(quality, allow_nrt=True) is nrt


def test_extraction_initializes_runtime_without_writable_home(tmp_path):
    import os
    import subprocess
    import sys
    env = {**os.environ, 'XDG_CONFIG_HOME': str(tmp_path / 'config'),
           'XDG_CACHE_HOME': str(tmp_path / 'cache')}
    result = subprocess.run([sys.executable, '-c', '''
from pathlib import Path
from unittest.mock import patch
with patch.object(Path, 'home', return_value=Path('/')):
    from argus_prophet.services.aia.extraction import extract_frame
    try:
        extract_frame('/nonexistent-aia-runtime-test.fits')
    except ValueError as exc:
        assert 'Did not find any files' in str(exc)
    else:
        raise AssertionError('Expected nonexistent FITS')
    import os
    import sunpy
    manager = Path(sunpy.config.get('downloads', 'remote_data_manager_dir'))
    assert manager.is_relative_to(Path(os.environ['XDG_CACHE_HOME']))
    assert Path(os.environ['XDG_CONFIG_HOME']).is_dir()
'''], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'will be ignored' not in result.stderr
