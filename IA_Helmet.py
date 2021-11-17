#!pip install imageai

import numpy as np
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
'''
# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict

#!wget "https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5"

path_folder_annotations = './hard-hat-detection/annotations'
path_folder_images = './hard-hat-detection/images'

path_annotations = []
for i in Path(path_folder_annotations).glob('*.xml'):
    path_annotations.append(i)
path_annotations = sorted(path_annotations)

path_images = []
for i in Path(path_folder_images).glob('*.png'):
    path_images.append(i)
path_images = sorted(path_images)

print(path_annotations[0])
# Open to visualize the XML
with open(path_annotations[0], 'r') as file:
    print(file.read())

# Testing function
print(extract_info_from_xml('./hard-hat-detection/annotations/hard_hat_workers0.xml'))

# Split data : 90% Train, 10% Test
path_train_annot = path_annotations[:4000]
path_test_annot = path_annotations[4000:4500]
path_val_annot = path_annotations[4500:5000]
path_train_images = path_images[:4000]
path_test_images = path_images[4000:4500]
path_val_images = path_images[4500:5000]

# Creating directories to put train/test data
os.makedirs('imageai/data/train/annotations',exist_ok = True)
os.makedirs('imageai/data/train/images', exist_ok = True)
os.makedirs('imageai/data/test/annotations', exist_ok = True)
os.makedirs('imageai/data/test/images', exist_ok = True)
os.makedirs('imageai/data/val/annotations', exist_ok = True)
os.makedirs('imageai/data/val/images', exist_ok = True)

for i, (path_annot, path_img) in enumerate(zip(path_train_annot, path_train_images)): 
    shutil.copy(path_img, 'imageai/data/train/images/' + path_img.parts[-1])
    shutil.copy(path_annot, 'imageai/data/train/annotations/' + path_annot.parts[-1])
    
for i, (path_annot, path_img) in enumerate(zip(path_test_annot, path_test_images)): 
    shutil.copy(path_img, 'imageai/data/test/images/' + path_img.parts[-1])
    shutil.copy(path_annot, 'imageai/data/test/annotations/' + path_annot.parts[-1])    

for i, (path_annot, path_img) in enumerate(zip(path_val_annot, path_val_images)): 
    shutil.copy(path_img, 'imageai/data/val/images/' + path_img.parts[-1])
    shutil.copy(path_annot, 'imageai/data/val/annotations/' + path_annot.parts[-1]) 

from imageai.Detection.Custom import DetectionModelTrainer

detector = DetectionModelTrainer()
detector.setModelTypeAsYOLOv3()
detector.setDataDirectory(data_directory="./imageai/data")
detector.setTrainConfig(object_names_array=["helmet","head","person"],
                       batch_size=8,
                       num_experiments=5,
                       train_from_pretrained_model="pretrained-yolov3.h5")

detector.trainModel()

from imageai.Detection.Custom import DetectionModelTrainer

detector = DetectionModelTrainer()
detector.setModelTypeAsYOLOv3()
detector.setDataDirectory(data_directory="./imageai/data/")
metrics = detector.evaluateModel(model_path="imageai/data/models/",
                                json_path="imageai/data/json/detection_config.json",
                                iou_threshold=0.2,
                                object_threshold=0.3,
                                nms_threshold=0.5)

#!wget 'https://mapa-da-obra-producao.s3.amazonaws.com/wp-content/uploads/2019/04/342.jpg'
'''
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("imageai/data/models/detection_model-ex-010--loss-0029.102.h5")
detector.setJsonPath("imageai/data/json/detection_config.json")
detector.loadModel()

path_folder_testes = './Testes'

for i in Path(path_folder_testes).glob('*.png'):
    image = Image.open(str(i))
    image = image.resize((416,416))
    image.save(str(i))

path_testes = sorted([i for i in Path(path_folder_testes).glob('*.png')])
print(path_testes)

path_detections = []

for i in range(len(path_testes)):
    path_detections.append('./Resultados/detections_' + str(i) + '.png')
print(path_detections)

for i in range(len(path_detections)):
    detections = detector.detectObjectsFromImage(minimum_percentage_probability=60,
                                                input_image = str(path_testes[i]),
                                                output_image_path = path_detections[i])
    print(path_detections[i])
    for detection in detections:  
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print('\n')

    #Image.open('detected.jpg')
