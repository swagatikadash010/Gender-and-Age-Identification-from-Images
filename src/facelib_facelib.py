import cv2
import math
import tqdm
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from facelib import AgeGenderEstimator
from facelib import FaceDetector, AgeGenderEstimator

face_detector = FaceDetector()
age_gender_detector = AgeGenderEstimator()


def get_tags_for_one_image(image_file):
    img = plt.imread(image_file)
    faces, boxes, scores, landmarks = face_detector.detect_align(img)
    if faces.tolist() != []:
        genders, ages = age_gender_detector.detect(faces)
        gender_tags = genders
        if "Male" in gender_tags and "Female" in gender_tags:
            val = "Contains both female and male"
        # elif "Hard" in gender_age["Gender"]:
        #    val =  "Hard to identify the gender"
        elif "Male" in gender_tags:
            val = "Male"
        elif "Female" in gender_tags:
            val = "Female"
        return val
    else:
        return "No face in this image"


if __name__ == "__main__":
    import sys

    image_files = sys.argv[1:]
    for image_file in tqdm.tqdm(image_files):
        print(image_file.split("/")[-1] + "," + get_tags_for_one_image(image_file))
