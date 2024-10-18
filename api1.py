
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import cv2
import numpy as np

from core import fn

IMAGEDIR = "images/"

app = FastAPI()

@app.post("/upload/")
async def create_upload_file(
    file : UploadFile = File(...)
):
    # input image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_shape = image.shape

    # Facelib
    predict_name, img_shape, len_boxes = fn(image)

    # save name
    processed_filename = f"{uuid.uuid4()}.jpg"
    processed_image_path = f"{IMAGEDIR}{processed_filename}"
    # result
    cv2.imwrite(processed_image_path, image)
    results = {
        "processed_filename" : processed_filename,
        "predict_name" : predict_name,
        "shape" : img_shape,
        "len_boxes" : len_boxes
    }
    return results


# img_name, predict_name, img_shape, len_boxes = fn(r"C:\Users\Acer\Desktop\FaceRecAPI\test_images\001.png")
