from __future__ import annotations
import io
import boto3
from botocore.client import Config
from .config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET

def s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

def ensure_bucket():
    client = s3()
    buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if MINIO_BUCKET not in buckets:
        client.create_bucket(Bucket=MINIO_BUCKET)

def put_bytes(key: str, data: bytes, content_type: str="application/octet-stream"):
    ensure_bucket()
    client = s3()
    client.put_object(Bucket=MINIO_BUCKET, Key=key, Body=data, ContentType=content_type)

def get_bytes(key: str) -> bytes:
    client = s3()
    obj = client.get_object(Bucket=MINIO_BUCKET, Key=key)
    return obj["Body"].read()
