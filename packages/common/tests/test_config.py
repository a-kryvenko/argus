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


def test_storage_configuration(monkeypatch, tmp_path):
    from common import config
    monkeypatch.setattr(config, '_config', None)
    monkeypatch.setenv('ARGUS_WORKDIR', str(tmp_path))
    (tmp_path / 'configs').mkdir()
    for name in ('project.yaml', 'models_registry.yaml'):
        (tmp_path / 'configs' / name).write_text('{}')
    monkeypatch.delenv('DATA_STORAGE', raising=False)
    for key, value in {
        'S3_ENDPOINT_URL': 'https://s3.example.com', 'S3_ACCESS_KEY_ID': 'dev-key',
        'S3_SECRET_ACCESS_KEY': 'dev-secret', 'S3_REGION': 'us-east-1',
    }.items():
        monkeypatch.setenv(key, value)
    assert config.get_config().s3 == {
        'endpoint': 'https://s3.example.com', 'key': 'dev-key',
        'secret': 'dev-secret', 'region': 'us-east-1',
    }
    assert config.get_config().data_storage == 'local'
    monkeypatch.setattr(config, '_config', None)
    monkeypatch.setenv('DATA_STORAGE', 's3')
    assert config.get_config().data_storage == 's3'
