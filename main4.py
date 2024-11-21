import cv2
import onnxruntime as ort
import boto3
from PIL import Image
from io import BytesIO
import aws_keys
import numpy as np
import matplotlib.pyplot as plt
import pickle 
# from keras.models import load_model


print("Import done!!!")

# Connect S3 with key (Got it's from ...)
s3 = boto3.client(
    "s3",
    region_name           = aws_keys.AWS_DEFAULT_REGION,
    aws_access_key_id     = aws_keys.AWS_ACCESS_KEY_ID,
    aws_secret_access_key = aws_keys.AWS_SECRET_ACCESS_KEY
    )

bucket_name = "model-test-211124-0920"

# # image
# response = s3.get_object(
#     Bucket = bucket_name,
#     Key = "ComfyUI_00016_.png"
# )
# img_bytes = response['Body'].read()

# img_array = np.frombuffer(img_bytes, np.uint8)
# img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.show()

# # embeddings_datasets.pickle
# response = s3.get_object(
#     Bucket = bucket_name,
#     Key = "embeddings_datasets.pickle"
# )
# model_bytes = response['Body'].read()
# obj = pickle.loads(model_bytes)
# print(obj)

# # le.pickle
# response = s3.get_object(
#     Bucket = bucket_name,
#     Key = "le.pickle"
# )
# le_pickle_bytes = response['Body'].read()
# obj = pickle.loads(le_pickle_bytes)
# print(obj)



# normal_model.h5


# w600k_r50.onnx
response = s3.get_object(
    Bucket = bucket_name,
    Key = "w600k_r50.onnx"
)
face_embedding_Bytes = response['Body'].read()
face_embedding = ort.InferenceSession(face_embedding_Bytes)
