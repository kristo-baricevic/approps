import os, boto3
from botocore.client import Config
from botocore.exceptions import ClientError

ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
# ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000")
REGION   = os.getenv("S3_REGION", "us-east-1")
BUCKET   = os.getenv("S3_BUCKET", "uploads")
AK       = os.getenv("S3_ACCESS_KEY", "minioadmin")
SK       = os.getenv("S3_SECRET_KEY", "minioadmin")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=AK,
    aws_secret_access_key=SK,
    region_name=REGION,
    config=Config(signature_version="s3v4", s3={"addressing_style":"path"}),
)

def ensure_bucket():
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)
