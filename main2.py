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
model_name   = "w600k_r50.onnx"

# Get image -> show it
response = s3.get_object(
    Bucket = bucket_name,
    Key = model_name
)

model_data = response['Body'].read()

session = ort.InferenceSession(model_data)
inputs = session.get_inputs()
img_path = r"C:\Users\Acer\Desktop\Coconut_eye\01.png"
img = cv2.imread(img_path)
img = cv2.resize(img, (112, 112))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.transpose(img, (2, 0, 1)).astype(np.float32)
img = img / 255.0
img = np.expand_dims(img, axis=0) 

outputs = session.run(
    None,{inputs[0].name:img}
)
print(outputs)