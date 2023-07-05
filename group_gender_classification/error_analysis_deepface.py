from deepface import DeepFace
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import torch

image_path = 'examples/fairface_dataset/val'
label_path = 'datasets/cleaned_fairface_label_val.csv'

classifier_results = {}
for file in os.listdir(image_path):
    objs = DeepFace.analyze(img_path = image_path + "/" + file, 
            actions = ['gender', 'race'], enforce_detection = False
    )
    name = "val/" + file
    classifier_results[name] = [objs[0]['dominant_gender'], objs[0]['dominant_race'], objs[0]['race']]
    
myKeys = list(classifier_results.keys())
myKeys.sort()

labels_df = pd.read_csv(label_path)
lines = len(labels_df['file'])

groundtruth_dict = {}
for i in range(lines):
    filename = labels_df['file'][i]
    gender_label = labels_df['gender'][i]
    race_label = labels_df['race'][i]
    
    groundtruth_dict[filename] = [gender_label, race_label]
    
groundtruth_keys = list(groundtruth_dict.keys())
groundtruth_keys.sort()

files = 0
correctGender = 0
correctRace = 0

# analysis
incorrectGender = {}
incorrectRace = {}
for filename in myKeys:
    files += 1
    if groundtruth_dict[filename][0] == classifier_results[filename][0]:
        correctGender += 1
    else: 
        incorrectGender[filename] = ["GT: " + groundtruth_dict[filename][0] + ", Predicted: " + classifier_results[filename][0]]
        
    if groundtruth_dict[filename][1] == classifier_results[filename][1]:
        correctRace += 1
    else: 
        incorrectRace[filename] = ["GT: " + groundtruth_dict[filename][1] + ", Predicted: " + classifier_results[filename][1] + ", Race Confidence: " + str(classifier_results[filename][2])]

print(str(correctGender/files) + " " + str(correctRace/files))

# view and print errors
with open('error_analysis_output_deepface.txt', 'w') as f:
    f.write(str(correctGender/files) + " " + str(correctRace/files))
    f.write('\n')
    for file in incorrectGender.keys():
        f.write(file + ", " + str(incorrectGender[file]) + "\n")

    for file in incorrectRace.keys():
        f.write(file + ", " + str(incorrectRace[file]) + "\n")
