import cv2
import os
import time
import numpy as np
from utils import *

def extract(string_line):
    line = string_line.replace(" ", "")
    line = line.split('=')
    
    return int(line[-1])




class YoloV3FromDarkNet:
    '''
    source Repos at:
    GitHub: https://github.com/fyr91/face_detection.git
    GitHub: https://github.com/sthanhng/yoloface
    '''
    def __init__(self,pretrained_weights_path,config_file_path,classes_file_path, confidence_threshold=0.9, iou= 0.7):
        self.confidence_threshold = confidence_threshold
        self.IOU = iou

        self.pretrained_weights_path = pretrained_weights_path
        self.config_file_path = config_file_path
        self.classes_file_path = classes_file_path
        self.net = None
        self.input_shape = [None, None, None] # height, width, channels
        self.classes=[]

        try:
            self.net = cv2.dnn.readNetFromDarknet(self.config_file_path, self.pretrained_weights_path)
            #backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            #targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        except Exception as e:
            print(e)
            exit(1)


        # get input shape from config file
        with open(self.config_file_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                if('width' in line.lower()):
                    self.input_shape[1] = extract(line)
                elif ('height' in line.lower()):
                    self.input_shape[0] = extract(line)
                elif ('channels' in line.lower()):
                    self.input_shape[2] = extract(line)
        
        # get classes from classes file
        with open(self.classes_file_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace(" ", "")
                self.classes.append(line)


        print("[info] input_shape =", self.input_shape)




    def detect(self, frame):
        h, w, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.input_shape[1], self.input_shape[0]), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        self.layers_names = self.net.getLayerNames()
        outs = self.net.forward(get_outputs_names(self.net))


        # Remove the bounding boxes with low confidence
        faces, confidences = post_process(frame, outs, self.confidence_threshold, self.IOU)

        return faces, confidences




