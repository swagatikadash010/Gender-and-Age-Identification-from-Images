#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import tqdm

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (200,200), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes



faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


padding=20

def get_tags_for_one_image(image_file):
    # read image
    frame = cv2.imread(image_file)
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        return image_file+","+"No face in this image"
    gender_age = {"Gender":[],"Age":[]}
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        #print ("GP",genderPreds)
        if abs(genderPreds[0][0]-genderPreds[0][1])<0.1:
            gender = "Hard"
        else:
            gender = genderList[genderPreds[0].argmax()]
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
    elif "Hard" in gender_age["Gender"]: 
        val =  "Hard to identify the gender"
    elif "Male" in gender_age["Gender"]:
        val = "Male"
    elif "Female" in gender_age["Gender"]:
        val = "Female"
    return image_file+","+val

    
if __name__=="__main__":
    import sys
    image_files = sys.argv[1:]
    for image_file in tqdm.tqdm(image_files):
        print (get_tags_for_one_image(image_file))
