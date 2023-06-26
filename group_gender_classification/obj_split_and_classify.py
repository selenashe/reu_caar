from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

directory = 'examples/testset/'
cropped_image_save_path = 'runs/cropped/'


def crop_faces(original_image_name, cropped_image_save_path):
    '''
    Take an image, detect faces using yolo8, and crop each face.
    '''
    # Load a model
    model = YOLO('yolov8n.pt')  # load an official model

    # Predict with the model
    results = model(f'{original_image_name}')  # predict on an image

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs

    box_amount = len(boxes)

    xyxy_coordinates = boxes.xyxy # Get xyxy coordinates of bounded boxes

    left, top, right, bottom = [], [], [], [] 

    for xyxy in xyxy_coordinates:
        xyxy = xyxy.numpy()
        left.append(xyxy[0])
        top.append(xyxy[1])
        right.append(xyxy[2])
        bottom.append(xyxy[3])
       # print(left, top, right, bottom)
        

    im = Image.open(f'{original_image_name}')
    im = im.convert('RGB')

    for i in range(box_amount):
        im_new = im.crop((left[i], top[i], right[i], bottom[i]))
        current_image_name = os.path.basename(os.path.normpath(original_image_name))
        im_new.save(f'{cropped_image_save_path}{current_image_name}_{i}.jpg')


def classify_gender(image_to_classify_path):
    '''
    Take an image as input, returns 1) predicted gender 2) predicted probability
    '''
    gender_net_pretrained = 'gender_net.caffemodel'
    gender_net_model_file='deploy_gender.prototxt'
    gender_net = cv2.dnn.readNetFromCaffe(gender_net_model_file, gender_net_pretrained)
    gender_list=['Male','Female']
    example_image = np.array(Image.open(image_to_classify_path))

    meanValues=(104, 117, 123)
    imgWidth=227
    imgHeight=227
    input_image= cv2.dnn.blobFromImage(example_image, 1, (imgWidth, imgHeight), meanValues)

    gender_net.setInput(input_image)
    preds = gender_net.forward()
    return gender_list[preds[0].argmax()], sorted(preds[0])[-1]


def main(original_image_name):
    crop_faces(original_image_name, cropped_image_save_path)
    current_image_name = os.path.basename(os.path.normpath(original_image_name))
    for file in os.listdir(cropped_image_save_path):
        if file.startswith(current_image_name):
            print(file, classify_gender(f'{cropped_image_save_path}{file}'))


with open('out.txt', 'a') as f:
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if os.path.isfile(file):
            print(main(file), file = f)
