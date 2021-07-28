# Detect age and gender using detectron fall back mechanism

#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
from mtcnn import MTCNN
import tqdm

import torch, torchvision
import detectron2
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

    
def highlightFace(detectron_model, frame_record):
    faceBoxes = []
    outputs = detectron_model(frame_record)

    boxes = outputs["instances"].pred_boxes
    pred_classes = outputs['instances'].pred_classes.cpu().tolist()
    pred_class_names = list(map(lambda x: class_names[x], pred_classes))
    
    frameHeight=frame_record.shape[0]
    frameWidth=frame_record.shape[1]

    for i, label in enumerate(pred_class_names):
        if label == "person":
            box = boxes[i]
            x1,y1,x2,y2 = box.tensor.tolist()[0]
            x1 = round(x1)
            y1 = round(y1)
            x2 = round(x2)
            y2 = round(y2)
            if (abs(y2-y1) > (frameHeight /10)) and (abs(x2-x1) > frameWidth /10):
                faceBoxes.append([x1,y1,x2,y2])
            
    return faceBoxes

# Load caffee models by hassner and levi            

ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

padding=20

# Detectron models
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"

detectron_predictor = DefaultPredictor(cfg)
class_names = MetadataCatalog.get("coco_2017_train").thing_classes

def get_tags_for_one_image(image_file):
    # read image
    frame = cv2.imread(image_file)
    faceBoxes=highlightFace(detectron_predictor, frame)
    if faceBoxes==[]:
        return image_file.split("/")[-1]+","+"No face in this image"
    gender_age = {"Gender":[],"Age":[]}
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        #if abs(genderPreds[0][0]-genderPreds[0][1])<0.1:
        #    gender = "Hard"
        gender_l = gender_age["Gender"]
        gender_l.append(gender)
        gender_age["Gender"] = gender_l

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        age_l = gender_age["Age"]
        age_l.append(age)
        gender_age["Age"] = age_l

    # use this later if age is needed
    # return gender_age
    if "Male" in gender_age["Gender"] and "Female" in gender_age["Gender"]:
        val =  "Contains both female and male"
    #elif "Hard" in gender_age["Gender"]: 
    #    val =  "Hard to identify the gender"
    elif "Male" in gender_age["Gender"]:
        val = "Male"
    elif "Female" in gender_age["Gender"]:
        val = "Female"
    return image_file.split("/")[-1]+","+val

    
if __name__=="__main__":
    import sys
    image_files = sys.argv[1:]
    for image_file in tqdm.tqdm(image_files):
        print (get_tags_for_one_image(image_file))
