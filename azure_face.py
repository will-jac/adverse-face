import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

# This key will serve all examples in this document.
KEY = "447c7a2d48784ac4bd8e60d90c5bfc8a"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://face-adverse-face.cognitiveservices.azure.com/"

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Detect a face in an image that contains a single face
single_face_image_path = r"attacks\no_box\resnet_batch_10_decoders_1_mode_naive_iters_15000_lr_001\images\1\Al_Gore_0001.png" # change this to the path of whichever attack image(s) you want to use
single_image_name = os.path.basename(single_face_image_path)
# We use detection model 3 to get better performance.
detected_faces = face_client.face.detect_with_stream(image = open(single_face_image_path, 'rb'), detection_model='detection_03')
if not detected_faces:
    raise Exception('No face detected from image {}'.format(single_image_name))

# Display the detected face ID in the first single-face image.
# Face IDs are used for comparison to faces (their IDs) detected in other images.
print('Detected face ID from', single_image_name, ':')
for face in detected_faces: print (face.face_id)
print()

# Save this ID for use in Find Similar
first_image_face_ID = detected_faces[0].face_id

# Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    
    return ((left, top), (right, bottom))

def drawFaceRectangles() :
# Download the image from the url
    img = Image.open(single_face_image_path)

# For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(getRectangle(face), outline='red')

# Display the image in the default image browser.
    img.show()

# Uncomment this to show the face rectangles.
drawFaceRectangles()