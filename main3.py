import cv2
import onnxruntime as ort
import boto3
from PIL import Image
from io import BytesIO
import aws_keys
import numpy as np

print("Import done!!!")

# Connect S3 with key (Got it's from ...)
s3 = boto3.client(
    "s3",
    region_name           = aws_keys.AWS_DEFAULT_REGION,
    aws_access_key_id     = aws_keys.AWS_ACCESS_KEY_ID,
    aws_secret_access_key = aws_keys.AWS_SECRET_ACCESS_KEY
    )
bucket_name = "model-test-211124-0920"

# List all file names
response = s3.list_objects_v2(Bucket = bucket_name)
files = []
for obj in response.get("Contents",[]):
    print(obj)
    print(obj['Key'])
    files.append(obj['Key'])
print(files)


# {
#     'Key': 'w600k_r50.onnx',
#     'LastModified': datetime.datetime(2024, 11, 21, 2, 21, 56, tzinfo=tzutc()),
#     'ETag': '"aeeb15bb031c45ab50f97ec07dd9b1db-11"',
#     'Size': 174383860,
#     'StorageClass': 'STANDARD'
# }




