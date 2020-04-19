# -*- coding: utf-8 -*-
# @Author: fyr91
# @Github: https://github.com/fyr91/face_detection.git
# @Date:   2019-10-22 15:05:15
# @Last Modified by: OmarSayedMostafa
# @GitHub: https://github.com/OmarSayedMostafa
# @Last Modified time: 2020-3-24 
import cv2
import dlib
import numpy as np
from imutils import face_utils
import box_utils
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import time 

from static import ultra_light_640_onnx_model_path, ultra_light_320_onnx_model_path, dlib_5_face_landmarks_path


class UltraLightOnnxFaceDetector:
    '''
    Ultra-light face detector by Linzaer and MobileFaceNet¹.
    Explained by the author at: https://towardsdatascience.com/real-time-face-recognition-with-cpu-983d35cc3ec5
    Original Repo at GitHub: https://github.com/fyr91/face_detection.git 
    '''

    def __init__(self, target_model=640, threshold=0.99):

        self.threshold= threshold
        self.target_model=target_model
        #---------------------------------------------------------
        if self.target_model==640:
            self.onnx_model_path = ultra_light_640_onnx_model_path
        else:
            self.onnx_model_path = ultra_light_320_onnx_model_path
        #---------------------------------------------------------
        # loading target onnx model 640x480 or 320x240
        self.onnx_model = onnx.load(self.onnx_model_path)
        self.predictor = prepare(self.onnx_model)
        self.ort_session = ort.InferenceSession(self.onnx_model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        #---------------------------------------------------------
        # load 5 point dlib shape predictor shape 
        # self.shape_predictor = dlib.shape_predictor(dlib_5_face_landmarks_path)
        # self.fa = face_utils.facealigner.FaceAligner(self.shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

        self.img_mean = np.array([127, 127, 127])

    def detect(self,frame):
        h, w, _ = frame.shape
        # preprocess img acquired
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
        if self.target_model==640:
            img = cv2.resize(img, (640, 480)) # resize
        else:
            img = cv2.resize(img, (320, 240)) # resize

        img = (img - self.img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = self.ort_session.run(None, {self.input_name: img})
        boxes, labels, probs = box_utils.predict(w, h, confidences, boxes, prob_threshold=self.threshold)
        
        return boxes, probs

