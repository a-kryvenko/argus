"""Run against dev SeaweedFS with ARGUS_S3_TEST_ENDPOINT=http://localhost:8333."""
from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
from urllib.request import urlopen
from urllib.error import HTTPError
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import pytest

from common.data import datamanager

pytestmark = pytest.mark.skipif(not os.getenv('ARGUS_S3_TEST_ENDPOINT'), reason='Set ARGUS_S3_TEST_ENDPOINT to test SeaweedFS')


@pytest.fixture
def storage(monkeypatch):
    endpoint = os.environ['ARGUS_S3_TEST_ENDPOINT']
    def client(reader=False):
        role = 'reader' if reader else 'writer'
        return boto3.client('s3', endpoint_url=endpoint, region_name='us-east-1',
            aws_access_key_id='argus-dev-' + role, aws_secret_access_key='argus-dev-' + role + '-secret',
            config=Config(s3={'addressing_style': 'path'}, connect_timeout=3, read_timeout=10))
    writer, reader = client(), client(True)
    prefix = 'integration/' + uuid4().hex
    monkeypatch.setattr(datamanager, 'get_config', lambda: SimpleNamespace(data_storage='s3'))
    monkeypatch.setattr(datamanager, 's3_client', writer)
    yield SimpleNamespace(writer=writer, reader=reader, endpoint=endpoint, key=prefix + '/image.npz')
    for bucket in ('argus-private', 'argus-public'):
        writer.delete_object(Bucket=bucket, Key=prefix + '/image.npz')
        writer.delete_object(Bucket=bucket, Key=prefix + '/race')


def test_real_s3_keeps_first_object_and_upload_time(storage):
    first = datamanager.put_immutable_file(b'first', 'argus-private', storage.key, content_type='application/octet-stream')
    second = datamanager.put_immutable_file(b'replacement', 'argus-private', storage.key, content_type='application/octet-stream')
    assert first == second
    assert datamanager.read_file('argus-private', storage.key) == b'first'
    assert storage.reader.get_object(Bucket='argus-private', Key=storage.key)['Body'].read() == b'first'
    with pytest.raises(ClientError) as error:
        storage.reader.put_object(Bucket='argus-private', Key=storage.key, Body=b'cannot write')
    assert error.value.response['ResponseMetadata']['HTTPStatusCode'] == 403


def test_anonymous_browser_can_only_read_public_objects(storage):
    for bucket in ('argus-private', 'argus-public'):
        datamanager.put_immutable_file(b'image', bucket, storage.key, content_type='image/webp')
    with urlopen(storage.endpoint + '/argus-public/' + storage.key, timeout=5) as response:
        assert response.read() == b'image'
        assert response.headers['Content-Type'] == 'image/webp'
        assert 'immutable' in response.headers['Cache-Control']
    with pytest.raises(HTTPError) as error:
        urlopen(storage.endpoint + '/argus-private/' + storage.key, timeout=5)
    assert error.value.code == 403


def test_conditional_create_has_one_winner_under_concurrent_uploads(storage):
    key = storage.key.rsplit('/', 1)[0] + '/race'
    def put(index):
        try:
            storage.writer.put_object(Bucket='argus-private', Key=key, Body=str(index).encode(), IfNoneMatch='*')
            return index
        except ClientError as error:
            assert error.response['ResponseMetadata']['HTTPStatusCode'] in (409, 412)
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        winners = [value for value in pool.map(put, range(16)) if value is not None]
    assert len(winners) == 1
    assert datamanager.read_file('argus-private', key) == str(winners[0]).encode()
