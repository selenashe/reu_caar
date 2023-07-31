from ultralytics import YOLO
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

directory = 'images/sd_generated'
cropped_image_save_path = 'images/cropped/'
# label_path = 'datasets/testset_labeled.csv'

def crop_faces(original_image_name, cropped_image_save_path):
    '''
    Take an image, detect faces using yolov8 (face), and crop each face.
    '''
    # Load a model
    model = YOLO('yolov8n-face.pt')  # load an official model

    # Predict with the model
    results = model(f'{original_image_name}')  # predict on an image

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs

    box_amount = len(boxes)

    xyxy_coordinates = boxes.xyxy # Get xyxy coordinates of bounded boxes

    left, top, right, bottom = [], [], [], [] 

    for xyxy in xyxy_coordinates:
        xyxy = xyxy.cpu().numpy()
        left.append(xyxy[0])
        top.append(xyxy[1])
        right.append(xyxy[2])
        bottom.append(xyxy[3])
        

    im = Image.open(f'{original_image_name}')
    im = im.convert('RGB')

    for i in range(box_amount):
        if boxes[i].conf > 0.5: # confirm face confidence > 0.5
            im_new = im.crop((left[i], top[i], right[i], bottom[i]))
            current_image_name = os.path.basename(os.path.normpath(original_image_name))
            if current_image_name.endswith('.jpg'):
                current_image_name = current_image_name.removesuffix('.jpg')

            im_new.save(f'{cropped_image_save_path}{current_image_name}_{i}.jpg')


def classify_images(cropped_path, classifier_output):
    '''
    For each image in directory, run DeepFace classifier and store output in row of dataframe.
    
        cropped_path: path to cropped images
        classifier_output: dataframe that stores classifier results
    '''
    
    classifier_results = {}
    idx = 0
    for file in os.listdir(cropped_path):
        objs = DeepFace.analyze(img_path = cropped_path + "/" + file, 
                actions = ['gender', 'race'], enforce_detection = False
        )

        name = file
        classifier_results[name] = [objs[0]['dominant_gender'], objs[0]['gender'], objs[0]['dominant_race'], objs[0]['race']]

        # set up row in df
        row = pd.DataFrame({"file": name,
                            "dominant_gender": [objs[0]['dominant_gender']], 
                            "Man" : [objs[0]['gender']['Man']],
                            "Woman" : [objs[0]['gender']['Woman']],
                            "dominant_race" : [objs[0]['dominant_race']],
                            "white": [objs[0]['race']['white']],
                            "latino hispanic": [objs[0]['race']['latino hispanic']],
                            "asian": [objs[0]['race']['asian']],
                            "black": [objs[0]['race']['black']],
                            "middle eastern": [objs[0]['race']['middle eastern']],
                            "indian": [objs[0]['race']['indian']]},                        
                            index = [idx])     
        classifier_output = pd.concat([classifier_output, row])
        idx += 1    
        
        # export results to csv
        classifier_output = classifier_output.sort_values(by = 'file')
        classifier_output.to_csv("output/pipeline_output.csv", index = False)        
        
def main():
    # crop images and store in path
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if os.path.isfile(file):
            crop_faces(file, cropped_image_save_path)
    
    # set up dataframe to store results
    classifier_output = pd.DataFrame(columns = ['file','dominant_gender', 'Man', 'Woman', 
                                 'dominant_race', 'white', 'latino hispanic', 'asian', 
                                 'black', 'middle eastern', 'indian'])
    
    classify_images(cropped_image_save_path, classifier_output)

main()
