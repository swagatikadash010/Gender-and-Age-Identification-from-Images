#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
from mtcnn import MTCNN
import tqdm
from facelib import AgeGenderEstimator
from facelib import FaceDetector, AgeGenderEstimator
import matplotlib.pyplot as plt

face_detector = FaceDetector()
age_gender_detector = AgeGenderEstimator()


def get_tags_for_one_image(image_file):
	img = plt.imread(image_file)
	faces, boxes, scores, landmarks = face_detector.detect_align(img)
	gender_age = {"Gender":[],"Age":[]}
	print (faces.shape)
	if faces.tolist()!=[]:
		genders, ages = age_gender_detector.detect(faces)
		gender_age["Gender"] = genders
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
	else:
		return image_file.split("/")[-1]+","+"No face in this image"

    
if __name__=="__main__":
    import sys
    image_files = sys.argv[1:]
    for image_file in tqdm.tqdm(image_files):
        print (get_tags_for_one_image(image_file))
