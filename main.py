# #upload/main.py
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import FileResponse
# import os
# from random import randint
# import uuid

# IMAGEDIR = "images/"

# app = FastAPI()

# @app.post("/upload/")
# async def create_upload_file(file: UploadFile = File(...)):
#     file.filename = f"{uuid.uuid4()}.jpg"
#     contents = await file.read()
#     #save the file
#     with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
#         f.write(contents)
#     return {"filename": file.filename}

# @app.get("/show/")
# async def read_random_file():
 
#     # get random file from the image directory
#     files = os.listdir(IMAGEDIR)
#     random_index = randint(0, len(files) - 1)
 
#     path = f"{IMAGEDIR}{files[random_index]}"
     
#     return FileResponse(path)



#############################################################################
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import FileResponse
# import os
# from random import randint
# import uuid
# import cv2
# import numpy as np

# IMAGEDIR = "images/"

# app = FastAPI()

# @app.post("/upload/")
# async def create_upload_file(
#     file : UploadFile = File(...)
# ):
#     # input image
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     img_shape = image.shape
#     # save name
#     processed_filename = f"{uuid.uuid4()}.jpg"
#     processed_image_path = f"{IMAGEDIR}{processed_filename}"
#     # result
#     cv2.imwrite(processed_image_path, image)
#     results = {
#         "processed_filename" : processed_filename,
#         "shape" : img_shape
#     }
#     return {"message": results}

#############################################################################
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import cv2
import numpy as np

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
    # save name
    processed_filename = f"{uuid.uuid4()}.jpg"
    processed_image_path = f"{IMAGEDIR}{processed_filename}"
    # result
    cv2.imwrite(processed_image_path, image)
    results = {
        "processed_filename" : processed_filename,
        "shape" : img_shape
    }
    return results
