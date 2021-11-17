#-----------------------------------------------Imports-----------------------------------------------------#

import numpy as np
import os
import shutil
import xml.etree.ElementTree as ET
import threading
import time
import sys

from pathlib import Path
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from imageai.Detection.Custom import CustomObjectDetection

#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------- Varaible Globals ----------------------------------------------#
images_detected = 0

#-----------------------------------------------------------------------------------------------------------#
def detect_Helmet():
    global images_detected
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("imageai/data/models/detection_model-ex-010--loss-0029.102.h5")
    detector.setJsonPath("imageai/data/json/detection_config.json")
    detector.loadModel()

    while(True):
        path_folder_testes = './Testes'

        for i in Path(path_folder_testes).glob('*.png'):
            image = Image.open(str(i))
            image = image.resize((416,416))
            image.save(str(i))

        path_testes = sorted([i for i in Path(path_folder_testes).glob('*.png')])
        print(path_testes)

        path_detections = []

        for i in range(len(path_testes)):
            path_detections.append('./Resultados/detections_' + str(i + images_detected) + '.png')
        images_detected += len(path_testes)
        #print(path_detections)

        for i in range(len(path_detections)):
            detections = detector.detectObjectsFromImage(minimum_percentage_probability=60, input_image = str(path_testes[i]), output_image_path = path_detections[i])
            
            print(path_detections[i])
            for detection in detections:  
                print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
                #if(detection["name"] == "head"):

            print('\n')
            os.remove(str(path_testes[i]))

        if (len(os.listdir(path_folder_testes))) == 0:
            print("No more images to treat...")
            print('Waiting 30s for more images...')
            time.sleep(30)
            if (len(os.listdir(path_folder_testes))) == 0:
                print("Not found more images, closing thread...")
                sys.exit()


threading.Thread(target=lambda : detect_Helmet()).start()