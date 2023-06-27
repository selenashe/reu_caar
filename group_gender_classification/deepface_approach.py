from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import os

cropped_image_save_path = 'runs/cropped_obj/'

results = {}
for file in os.listdir(cropped_image_save_path):
    objs = DeepFace.analyze(img_path = cropped_image_save_path + "/" + file, 
            actions = ['gender'], enforce_detection = False
    )
    results[file] = objs[0]['dominant_gender']

myKeys = list(results.keys())
myKeys.sort()
for i in myKeys:
    print(i + ": " + results[i] +"\n")
sorted_dict = {i: results[i] for i in myKeys}
