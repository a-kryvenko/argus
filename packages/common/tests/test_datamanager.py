import io
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from PIL import Image
import pytest

from common.data import datamanager as manager


@pytest.fixture
def client(monkeypatch):
    client = Mock()
    monkeypatch.setattr(manager, 's3_client', client)
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(data_storage='s3'))
    return client


@pytest.mark.parametrize('endpoint,region', [
    ('https://s3.example.com', 'us-east-1'),
    ('https://account.r2.cloudflarestorage.com', 'auto'),
])
def test_client_uses_configured_s3_and_is_reused(monkeypatch, endpoint, region):
    factory = Mock()
    monkeypatch.setattr(manager, 's3_client', None)
    monkeypatch.setattr(manager.boto3, 'client', factory)
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(
        s3={'endpoint': endpoint, 'key': 'key', 'secret': 'secret', 'region': region}))
    assert manager._get_client() is manager._get_client()
    factory.assert_called_once()
    assert factory.call_args.args == ('s3',)
    arguments = factory.call_args.kwargs
    assert arguments['endpoint_url'] == endpoint
    assert arguments['aws_access_key_id'] == 'key'
    assert arguments['aws_secret_access_key'] == 'secret'
    assert arguments['region_name'] == region
    assert arguments['config'].s3 == {'addressing_style': 'path'}


def test_missing_credentials_fail_before_creating_client(monkeypatch):
    monkeypatch.setattr(manager, 's3_client', None)
    factory = Mock()
    monkeypatch.setattr(manager.boto3, 'client', factory)
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(s3={}))
    with pytest.raises(RuntimeError, match='not configured'):
        manager._get_client()
    factory.assert_not_called()


def test_mp4_upload_matches_s3_contract(monkeypatch):
    client = boto3.client('s3', region_name='us-east-1',
                          aws_access_key_id='test', aws_secret_access_key='test')
    monkeypatch.setattr(manager, 's3_client', client)
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(data_storage='s3'))
    with Stubber(client) as stub:
        stub.add_response('put_object', {}, {
            'Bucket': 'media', 'Key': 'sdo/171/movie.mp4', 'Body': b'video',
            'ContentType': 'video/mp4', 'CacheControl': manager.IMMUTABLE_CACHE})
        assert manager.upload_file(b'video', 'media', '/sdo/171/', '/movie.mp4/') == 'sdo/171/movie.mp4'
        stub.assert_no_pending_responses()


def test_responsive_images_preserve_original_aspect_ratio_and_transparency(client):
    source = io.BytesIO()
    Image.new('RGBA', (80, 40), (255, 0, 0, 0)).save(source, format='PNG')
    keys = manager.upload_responsive_image(source.getvalue(), 'media', 'sdo', '171.png', [
        manager.ImageVariant('web', 40, 'WEBP'),
        manager.ImageVariant('jpeg', 20),
        manager.ImageVariant('large', 160, 'WEBP'),
    ])
    objects = {call.kwargs['Key']: call.kwargs for call in client.put_object.call_args_list}
    assert objects[keys['original']]['Body'] == source.getvalue()
    for label, size, format in [('web', (40, 20), 'WEBP'),
                                ('jpeg', (20, 10), 'JPEG'), ('large', (80, 40), 'WEBP')]:
        with Image.open(io.BytesIO(objects[keys[label]]['Body'])) as image:
            assert image.size == size
            assert image.format == format
            if format == 'WEBP':
                assert image.getpixel((0, 0))[3] == 0
    assert objects[keys['web']]['ContentType'] == 'image/webp'
    assert objects[keys['jpeg']]['ContentType'] == 'image/jpeg'


@pytest.mark.parametrize('variants', [
    [manager.ImageVariant('bad', 0)],
    [manager.ImageVariant('original', 20)],
    [manager.ImageVariant('same', 20), manager.ImageVariant('same', 40)],
    [manager.ImageVariant('bad', 20, 'unknown')],
])
def test_invalid_variants_upload_nothing(client, variants):
    with pytest.raises(ValueError):
        manager.upload_responsive_image(b'invalid', 'media', variants=variants)
    client.put_object.assert_not_called()


def test_upload_failure_propagates(client):
    client.put_object.side_effect = ClientError(
        {'Error': {'Code': 'AccessDenied', 'Message': 'denied'}}, 'PutObject')
    with pytest.raises(ClientError):
        manager.upload_file(b'video', 'media', name='movie.mp4')


def test_local_upload_never_contacts_s3(client, monkeypatch, tmp_path):
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(
        data_storage='local', data_root=tmp_path,
        s3={'endpoint': 'https://production.example.com', 'key': 'key', 'secret': 'secret'}))
    key = manager.upload_file(b'video', 'media', '/sdo/171/', '/movie.mp4/')
    assert key == 'sdo/171/movie.mp4'
    path = tmp_path / 'media' / 'media' / key
    assert path.read_bytes() == b'video'
    manager.upload_file(b'updated', 'media', 'sdo/171', 'movie.mp4')
    assert path.read_bytes() == b'updated'
    assert list(path.parent.iterdir()) == [path]
    client.put_object.assert_not_called()


def test_local_responsive_images(client, monkeypatch, tmp_path):
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(
        data_storage='local', data_root=tmp_path))
    source = io.BytesIO()
    Image.new('RGB', (80, 40), 'red').save(source, format='JPEG')
    keys = manager.upload_responsive_image(source.getvalue(), 'media', name='171.jpg',
        variants=[manager.ImageVariant('web', 40, 'WEBP')])
    root = tmp_path / 'media' / 'media'
    assert (root / keys['original']).read_bytes() == source.getvalue()
    with Image.open(root / keys['web']) as image:
        assert image.format == 'WEBP'
        assert image.size == (40, 20)
    client.put_object.assert_not_called()


@pytest.mark.parametrize('bucket,key', [('media', '../escape.mp4'), ('../escape', 'movie.mp4')])
def test_local_upload_rejects_escape(client, monkeypatch, tmp_path, bucket, key):
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(
        data_storage='local', data_root=tmp_path))
    with pytest.raises(ValueError):
        manager.upload_file(b'video', bucket, name=key)
    assert not list(tmp_path.iterdir())


def test_local_upload_rejects_symlink_escape(client, monkeypatch, tmp_path):
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(
        data_storage='local', data_root=tmp_path))
    root = tmp_path / 'media' / 'media'
    root.mkdir(parents=True)
    (root / 'outside').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        manager.upload_file(b'video', 'media', 'outside', 'movie.mp4')
    assert not (tmp_path / 'movie.mp4').exists()
