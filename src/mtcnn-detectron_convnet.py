import cv2
import math
import tqdm
import os, pathlib
import torch, torchvision
import detectron2
import numpy as np
import json, cv2, random

from mtcnn import MTCNN
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

current_path = os.getcwd()
project_path = pathlib.Path(current_path).parent.absolute()

# INITIALIZE MODELS
genderProto = str(project_path) + "/models/gender_deploy.prototxt"
genderModel = str(project_path) + "/models/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ["Male", "Female"]
genderNet = cv2.dnn.readNet(genderModel, genderProto)
padding = 20

faceNet = MTCNN()
# Detectron models
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.DEVICE = "cpu"
detectron_predictor = DefaultPredictor(cfg)
class_names = MetadataCatalog.get("coco_2017_train").thing_classes


def highlightFace(net, frame_record, conf_threshold=0.1):
    frame = cv2.cvtColor(frame_record, cv2.COLOR_BGR2RGB)

    # This is done through the multitask model MTCNN
    facedata = net.detect_faces(frame)
    faceBoxes = []
    for face in facedata:
        x1, y1, width, height = face["box"]
        # confidence = face["confidence"]
        # if confidence > conf_threshold:
        x2 = x1 + width
        y2 = y1 + height
        faceBoxes.append([x1, y1, x2, y2])
    return faceBoxes


def fallback_highlight_person(detectron_model, frame_record):
    faceBoxes = []
    outputs = detectron_model(frame_record)

    boxes = outputs["instances"].pred_boxes
    pred_classes = outputs["instances"].pred_classes.cpu().tolist()
    pred_class_names = list(map(lambda x: class_names[x], pred_classes))

    frameHeight = frame_record.shape[0]
    frameWidth = frame_record.shape[1]

    for i, label in enumerate(pred_class_names):
        if label == "person":
            box = boxes[i]
            x1, y1, x2, y2 = box.tensor.tolist()[0]
            x1 = round(x1)
            y1 = round(y1)
            x2 = round(x2)
            y2 = round(y2)
            if (abs(y2 - y1) > (frameHeight / 10)) and (abs(x2 - x1) > frameWidth / 10):
                faceBoxes.append([x1, y1, x2, y2])
    return faceBoxes


def get_tags_for_one_image(image_file):
    # read image
    frame = cv2.imread(image_file)
    faceBoxes = highlightFace(faceNet, frame)
    if faceBoxes == []:
        # fallback to detectron based recognition
        faceBoxes = fallback_highlight_person(detectron_predictor, frame)
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
