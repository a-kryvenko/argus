"""Store source files and responsive image variants locally or in S3."""
import io
import mimetypes
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import boto3
from botocore.config import Config
from common.config import get_config
from PIL import Image, ImageOps

IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
IMAGE_FORMATS = {
    "JPEG": ("jpg", "image/jpeg"),
    "WEBP": ("webp", "image/webp"),
    "PNG": ("png", "image/png")
}


@dataclass(frozen=True)
class ImageVariant:
    name: str
    width: int
    format: str = "JPEG"
    quality: int = 82

s3_client = None


def _get_client():
    global s3_client
    
    if s3_client is None:
        s3_conf = get_config().s3
        if not all(s3_conf.get(field) for field in ("endpoint", "key", "secret")):
            raise RuntimeError("S3 access not configured")

        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_conf["endpoint"],
            aws_access_key_id=s3_conf["key"],
            aws_secret_access_key=s3_conf["secret"],
            region_name=s3_conf["region"],
            config=Config(s3={"addressing_style": "path"}),
        )
    
    return s3_client


def upload_responsive_image(
    file: bytes,
    bucket: str,
    prefix: str = "",
    name: str = "",
    variants: list[ImageVariant] | None = None,
) -> dict[str, str]:
    """Return keys for the unchanged original and named variants after all uploads."""

    if variants is None:
        variants = []
    
    labels = {"original"}
    for variant in variants:
        if (
            not variant.name
            or variant.name in labels
            or "/" in variant.name
            or variant.width < 1
            or not 1 <= variant.quality <= 100
            or variant.format.upper() not in IMAGE_FORMATS
        ):
            raise ValueError("Invalid or duplicate image variant")
        
        labels.add(variant.name)

    name = name.strip("/")
    if name:
        stem = str(PurePosixPath(name).with_suffix(""))
    else:
        stem = uuid.uuid4().hex
    
    prepared = []
    with Image.open(io.BytesIO(file)) as source:
        source.load()

        if source.format not in IMAGE_FORMATS:
            raise ValueError("Expected a JPEG, PNG or WebP source image")
        
        extension, content_type = IMAGE_FORMATS[source.format]

        prepared.append(("original", f"{stem}.{extension}", file, content_type))
        image = ImageOps.exif_transpose(source)

        for variant in variants:
            format = variant.format.upper()
            extension, content_type = IMAGE_FORMATS[format]

            width = min(variant.width, image.width)
            height = max(1, round(image.height * width / image.width))

            resized = image.resize((width, height), Image.Resampling.LANCZOS)
            resized = resized.convert("RGB" if format == "JPEG" else "RGBA")

            output = io.BytesIO()
            resized.save(output, format=format, quality=variant.quality)

            prepared.append((
                variant.name,
                f"{stem}_{variant.name}.{extension}",
                output.getvalue(),
                content_type
            ))
    
    result = {}

    for label, filename, content, content_type in prepared:
        result[label] = upload_file(content, bucket, prefix, filename, content_type=content_type)

    return result


def upload_file(
    file: bytes,
    bucket: str,
    prefix: str = "",
    name: str = "",
    *,
    content_type: str | None = None,
    cache_control: str = IMMUTABLE_CACHE,
) -> str:
    """Return the object key. Use unique/versioned names with immutable caching."""
    name = name.strip("/") or uuid.uuid4().hex
    prefix = prefix.strip("/")
    key = f"{prefix}/{name}" if prefix else name

    storage = get_config().data_storage

    if storage == "local":
        _upload_local(file, bucket, key)
    elif storage == "s3":
        _upload_s3(
            file,
            bucket,
            key,
            content_type or mimetypes.guess_type(name)[0] or "application/octet-stream",
            cache_control
        )
    else:
        raise ValueError(f"Unknown storage: {storage}")
    
    return key


def _upload_local(content: bytes, bucket: str, key: str) -> None:
    if not bucket or bucket in {".", ".."} or "/" in bucket:
        raise ValueError("Invalid bucket name")
    
    root = (get_config().data_root / "media").resolve()
    bucket_root = root / bucket
    path = (bucket_root / key).resolve()

    if not path.is_relative_to(bucket_root) or path == bucket_root:
        raise ValueError("File path is outside its bucket")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _upload_s3(
        content: bytes,
        bucket: str,
        key: str,
        content_type: str,
        cache_control: str
) -> None:
    _get_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=content,
        ContentType=content_type,
        CacheControl=cache_control,
    )
