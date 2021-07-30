# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import json
import sys
import tqdm

client = boto3.client("rekognition")
s3 = boto3.resource("s3")


def detect_faces(photo, bucket):
    response = client.detect_faces(
        Image={"S3Object": {"Bucket": bucket, "Name": photo}}, Attributes=["ALL"]
    )
    faceDetails = response["FaceDetails"]
    if len(faceDetails) == 0:
        return "No face in this image"
    gender_tags = []
    for faceDetail in faceDetails:
        gender_tags.append(str(faceDetail["Gender"]["Value"]))
    if "Male" in gender_tags and "Female" in gender_tags:
        val = "Contains both female and male"
    # elif "Hard" in gender_age["Gender"]:
    #    val =  "Hard to identify the gender"
    elif "Male" in gender_tags:
        val = "Male"
    elif "Female" in gender_tags:
        val = "Female"
    return val


def main():
    dir_name = sys.argv[1]
    bucket = "swagibucket"
    for fn_details in tqdm.tqdm(s3.Bucket(bucket).objects.all()):
        file_name = fn_details.key
        if dir_name in file_name:
            if "jpg" in file_name or "png" in file_name:
                file_tag = file_name.replace(dir_name + "/", "")
                print(file_tag + "," + detect_faces(file_name, bucket))


if __name__ == "__main__":
    main()
