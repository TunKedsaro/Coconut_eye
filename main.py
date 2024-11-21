import boto3
from PIL import Image
from io import BytesIO
import aws_keys

print("Import done!!!")

# Connect S3 with key (Got it's from ...)
s3 = boto3.client(
    "s3",
    region_name = aws_keys.AWS_DEFAULT_REGION,
    aws_access_key_id = aws_keys.AWS_ACCESS_KEY_ID,
    aws_secret_access_key = aws_keys.AWS_SECRET_ACCESS_KEY
    )
bucket_name = "model-test-211124-0920"
image_key   = "ComfyUI_00016_.png"


# list all file in bucket
response = s3.list_objects_v2(
    Bucket = bucket_name
)
if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])
else:
    print("No file in your bucket Kub 555 -_-")


# Get image -> show it
response = s3.get_object(
    Bucket = bucket_name,
    Key = image_key
)
# Read and display the image
image_data = response['Body'].read()
image = Image.open(BytesIO(image_data))
image.show()