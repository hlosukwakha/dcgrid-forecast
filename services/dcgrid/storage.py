from __future__ import annotations

import os
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "dcgrid")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")


def _s3():
    # MinIO is S3-compatible. For local clusters, keep it simple.
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=MINIO_REGION,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket() -> None:
    s3 = _s3()
    try:
        s3.head_bucket(Bucket=MINIO_BUCKET)
        return
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        # MinIO commonly returns 404/NoSuchBucket/NotFound when missing
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise

    # Create bucket (LocationConstraint is optional; for us-east-1 omit it)
    try:
        if MINIO_REGION and MINIO_REGION not in ("us-east-1", "US", "us-east-1a"):
            s3.create_bucket(
                Bucket=MINIO_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": MINIO_REGION},
            )
        else:
            s3.create_bucket(Bucket=MINIO_BUCKET)
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise


def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    ensure_bucket()
    s3 = _s3()
    s3.put_object(Bucket=MINIO_BUCKET, Key=key, Body=data, ContentType=content_type)


def get_bytes(key: str) -> bytes:
    s3 = _s3()
    obj = s3.get_object(Bucket=MINIO_BUCKET, Key=key)
    return obj["Body"].read()


def exists(key: str) -> bool:
    s3 = _s3()
    try:
        s3.head_object(Bucket=MINIO_BUCKET, Key=key)
        return True
    except ClientError:
        return False
