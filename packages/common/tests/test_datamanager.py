import io
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from PIL import Image
import pytest

from common.data import datamanager as manager
from datetime import UTC, datetime


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

def test_immutable_s3_upload_uses_conditional_create_and_recovers_existing_object(monkeypatch):
    client = boto3.client('s3', region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test')
    monkeypatch.setattr(manager, 's3_client', client)
    monkeypatch.setattr(manager, 'get_config', lambda: SimpleNamespace(data_storage='s3'))
    uploaded = datetime(2026, 11, 2, 10, tzinfo=UTC)
    location = dict(Bucket='argus-private', Key='images/SDO/aia193/202611021000/512.npz')
    with Stubber(client) as stub:
        stub.add_response('put_object', {}, {**location, 'Body': b'first', 'IfNoneMatch': '*',
            'ContentType': 'application/octet-stream', 'CacheControl': 'private'})
        stub.add_response('head_object', {'LastModified': uploaded}, location)
        stub.add_client_error('put_object', service_error_code='PreconditionFailed', http_status_code=412,
            expected_params={**location, 'Body': b'revision', 'IfNoneMatch': '*',
                'ContentType': 'application/octet-stream', 'CacheControl': 'private'})
        stub.add_response('head_object', {'LastModified': uploaded}, location)
        for content in (b'first', b'revision'):
            assert manager.put_immutable_file(content, location['Bucket'], location['Key'],
                                               content_type='application/octet-stream') == uploaded
        stub.assert_no_pending_responses()


def test_immutable_s3_permission_failure_does_not_report_success(client):
    client.put_object.side_effect = ClientError(
        {'Error': {'Code': 'AccessDenied'}, 'ResponseMetadata': {'HTTPStatusCode': 403}}, 'PutObject')
    with pytest.raises(ClientError):
        manager.put_immutable_file(b'image', 'argus-public', 'images/AIA/193/202611021000/original.jpg',
                                  content_type='image/jpeg')
    client.head_object.assert_not_called()


def test_bounded_read_rejects_oversized_object_and_closes_stream(client):
    stream = io.BytesIO(b'123456789')
    client.get_object.return_value = {'Body': stream}
    with pytest.raises(ValueError, match='size limit'):
        manager.read_file('argus-private', 'image.fits', max_bytes=4)
    assert stream.closed


def test_temporary_source_file_is_removed_after_processing(monkeypatch):
    from uuid import uuid4
    reference = dict(id=uuid4(), timestamp=datetime.now(UTC), bucket='argus-private', key='image.fits')
    monkeypatch.setattr(manager, 'read_file', Mock(return_value=b'fits'))
    with pytest.raises(RuntimeError):
        with manager.temporary_file(reference) as path:
            assert path.read_bytes() == b'fits'
            raise RuntimeError('processing failed')
    assert not path.exists() and not path.parent.exists()
