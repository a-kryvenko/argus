from common.config import _find_workdir


def test_installed_package_uses_explicit_workdir(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_WORKDIR', str(tmp_path))
    assert _find_workdir() == tmp_path


def test_config_discovery_from_nested_application_directory(monkeypatch, tmp_path):
    monkeypatch.delenv('ARGUS_WORKDIR', raising=False)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/project.yaml').write_text('{}')
    nested = tmp_path / 'apps/api'
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert _find_workdir() == tmp_path
