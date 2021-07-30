# A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import tqdm
import torch
import os, pathlib

from mtcnn import MTCNN
from facelib import AgeGenderEstimator

# INITIALIZE MODELS

current_path = os.getcwd()
project_path = pathlib.Path(current_path).parent.absolute()

faceProto = str(project_path) + "/models/opencv_face_detector.pbtxt"
faceModel = str(project_path) + "/models/opencv_face_detector_uint8.pb"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

faceNet = cv2.dnn.readNet(faceModel, faceProto)
padding = 20
age_gender_detector = AgeGenderEstimator()


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (200, 200), [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frameHeight / 150)),
                8,
            )
    return frameOpencvDnn, faceBoxes


def get_tags_for_one_image(image_file):
    # read image
    frame = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    resultImg, faceBoxes = highlightFace(faceNet, frame)
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
            face, 1.0, (112, 112), MODEL_MEAN_VALUES, swapRB=False
        )
        blob_facelib = blob.transpose(0, 2, 3, 1)
        genders, ages = age_gender_detector.detect(torch.Tensor(blob_facelib))
        gender_tags += genders

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
