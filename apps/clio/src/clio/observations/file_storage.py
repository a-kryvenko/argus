"""Physical object storage shared by all Clio file observations."""
from botocore.exceptions import BotoCoreError, ClientError

from common.data.datamanager import create_immutable_file, read_file
from clio.observations import files

PRIVATE_BUCKET = 'argus-private'
PUBLIC_BUCKET = 'argus-public'


def upload(content, bucket, key, *, content_type):
    existing = files.find(bucket=bucket, key=key)
    if existing:
        return existing[0], False
    uploaded, created = create_immutable_file(content, bucket, key, content_type=content_type)
    return files.register(timestamp=uploaded, bucket=bucket, key=key), created


def publish(content, bucket, key, *, content_type):
    return upload(content, bucket, key, content_type=content_type)[0]


def read(reference):
    try:
        return read_file(reference['bucket'], reference['key'])
    except FileNotFoundError:
        raise OSError('Registered observation object is unavailable') from None
    except (BotoCoreError, ClientError) as exc:
        raise OSError('Observation object storage is unavailable') from exc
