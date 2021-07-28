#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
from mtcnn import MTCNN
import tqdm

def highlightFace(face_cascade, frame):
    # This is done through the legacy harcascade models
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facedata = face_cascade.detectMultiScale(gray, 1.1, 4)
    faceBoxes = []
    for face in facedata:
        x1,y1,width,height = face
        x2 = x1+width
        y2 = y1+height
        faceBoxes.append([x1,y1,x2,y2])
    return faceBoxes
            
# Load caffee models by hassner and levi
# And also facecascade model from opencv for face detection 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
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

def get_tags_for_one_image(image_file):
    # read image
    frame = cv2.imread(image_file)
    faceBoxes=highlightFace(face_cascade,frame)
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
