# A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import tqdm
import os, pathlib
import matplotlib.pyplot as plt

from mtcnn import MTCNN
from facelib import FaceDetector


current_path = os.getcwd()
project_path = pathlib.Path(current_path).parent.absolute()

# INITIALIZE MODELS
face_detector = FaceDetector()

genderProto = str(project_path) + "/models/gender_deploy.prototxt"
genderModel = str(project_path) + "/models/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ["Male", "Female"]
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 20


def highlightFace(facelib_face_detector, image_file):
    img = plt.imread(image_file)
    img = img[:,:,:3]
    faces, boxes, scores, landmarks = facelib_face_detector.detect_align(img)
    faceBoxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)
        faceBoxes.append([x1, y1, x2, y2])
    return faceBoxes


def get_tags_for_one_image(image_file):
    # read image
    frame = cv2.imread(image_file)
    faceBoxes = highlightFace(face_detector, image_file)
    if faceBoxes == []:
        return "No face in this image"
    gender_tags = []
    for faceBox in faceBoxes:
        face = frame[
            max(0, faceBox[1] - padding) : min(
                faceBox[3] + padding, frame.shape[0] - 1
            ),
            max(0, faceBox[0] - padding) : min(
                faceBox[2] + padding, frame.shape[1] - 1
            ),
        ]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # if abs(genderPreds[0][0]-genderPreds[0][1])<0.1:
        #    gender = "Hard"
        gender_tags.append(gender)

    if "Male" in gender_tags and "Female" in gender_tags:
        val = "Contains both female and male"
    # elif "Hard" in gender_age["Gender"]:
    #    val =  "Hard to identify the gender"
    elif "Male" in gender_tags:
        val = "Male"
    elif "Female" in gender_tags:
        val = "Female"
    return val


if __name__ == "__main__":
    import sys

    image_files = sys.argv[1:]
    for image_file in tqdm.tqdm(image_files):
        print(image_file.split("/")[-1] + "," + get_tags_for_one_image(image_file))
