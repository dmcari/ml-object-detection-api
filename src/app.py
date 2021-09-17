# Libraries for API execution
import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# Libraries for processing images
import cv2
from prediction import detect_and_draw_box

# Library for directories setting
import os

# Output directory with bounding boxes
dir_name = "images_uploaded"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

# API

# Assign an instance of the FastAPI class to the variable "app".
app = FastAPI(title="ML Model Deploy for Image Object Detection")

# List available models using Enum for convenience.
class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


# Allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Head over to http://localhost:8000/docs."


# This endpoint handles all the logic for the object detection to work.
# Contains the model and the image in which to perform object detection.
@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):

    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(
            status_code=415, detail="Unsupported file provided."
        )

    # 2. TRANSFORM RAW IMAGE INTO CV2 image

    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())

    # Start the stream from the beginning (position zero)
    image_stream.seek(0)

    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 3. RUN OBJECT DETECTION MODEL

    output_image = detect_and_draw_box(
        image, model="yolov3-tiny", confidence=0.5
    )

    # Save it in a folder within the server
    cv2.imwrite(f"images_uploaded/{filename}", output_image)

    # 4. STREAM THE RESPONSE BACK TO THE CLIENT

    # Open the saved image for reading in binary mode
    file_image = open(f"images_uploaded/{filename}", mode="rb")

    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg")


# Allows the server to be run in interactive environment
nest_asyncio.apply()

host = "0.0.0.0"

# Spin up the server
uvicorn.run(app, host=host, port=8000)
