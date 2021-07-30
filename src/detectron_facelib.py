import cv2
import math
import tqdm

import torch, torchvision
import detectron2
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from mtcnn import MTCNN
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from facelib import AgeGenderEstimator

# INITIALIZE MODELS

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20

# Detectron models for person detection (fallback model)
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

age_gender_detector = AgeGenderEstimator()


def highlightFace(detectron_model, frame_record):
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
    faceBoxes = highlightFace(detectron_predictor, frame)
    if faceBoxes == []:
        # fallback to detectron based recognition
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
