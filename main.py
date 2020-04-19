import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import atexit
import cv2
import time
from OpenCVDnn import opencvDNNFaceDetector
from openCVHaarCascade import CascadeFrontalFace
from dlibFaceDetection import dlibFaceDetector
from mtcnnFaceDetector import MTCNNDetector
from yoloV3FaceDetection import YoloV3FromDarkNet
from ultra_light_Detector import UltraLightOnnxFaceDetector
from utils import draw_predict
from faceRecogintion import FaceRecognition



#-----------------------------------------------------------------------------------------------------
def test(detector,recogonizer, calculate_fps=True):
    video_capture = cv2.VideoCapture(0)
    if calculate_fps:
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        frames_captured =0
        start_time = time.time()
        while frames_captured<fps:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Return's list of bounding box co-ordinates
            bboxes, confidences = detector(frame)

            frames_captured +=1

            for [left, top, right, bottom], confidence in zip(bboxes,confidences):
                draw_predict(frame, str(confidence), left, top, right, bottom)

            
            # Display the resulting image

            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        
        end_time = time.time()
        video_capture.release()
        # cv2.destroyAllWindows()

        seconds = end_time - start_time
        print("Time taken : {0} seconds".format(seconds))
        print("captured frames", frames_captured)
        estimated_fps  = frames_captured / seconds
        print("Estimated frames per second : {0}".format(estimated_fps))

        del start_time
        del end_time

        return estimated_fps




fps = {}

# print('\n\n\n\n HAAR CASCADE \n\n\n\n')
# opencv_HaarCascade_Detector = CascadeFrontalFace(threshold=0.99)
# fps['opencv_HaarCascade_Detector'] = test(opencv_HaarCascade_Detector.detect)
# del opencv_HaarCascade_Detector


# print('\n\n\n\n TF OPENCV DNN \n\n\n\n')
# opencv_TF_DnnDetector = opencvDNNFaceDetector(target_model='tf', threshold=0.99)
# fps['opencv_TF_DnnDetector'] = test(opencv_TF_DnnDetector.detect)
# del opencv_TF_DnnDetector



# print('\n\n\n\n opencv_CAFFE_DnnDetector \n\n\n\n')
# opencv_caffe_DnnDetector = opencvDNNFaceDetector(target_model='caffe', threshold=0.99)
# fps['opencv_caffe_DnnDetector'] = test(opencv_caffe_DnnDetector.detect)
# del opencv_caffe_DnnDetector




# print('\n\n\n\n DLIB HOG \n\n\n\n')
# dlib_Hog_Detector = dlibFaceDetector(target_method='Hog',scale_factor=0.5)
# fps['dlib_Hog_Detector'] = test(dlib_Hog_Detector.detect)
# del dlib_Hog_Detector



# print('\n\n\n\n DLIB MMOD \n\n\n\n')
# dlib_MMOD_Detector = dlibFaceDetector(target_method='MMOD',scale_factor=0.5)
# fps['dlib_MMOD_Detector'] = test(dlib_MMOD_Detector.detect)
# del dlib_MMOD_Detector


# print('\n\n\n\n MTCNN \n\n\n\n')
# mtcnnDetector = MTCNNDetector(threshold=0.99)
# fps['mtcnnDetector'] = test(mtcnnDetector.detect)
# del mtcnnDetector


faceRecognition = FaceRecognition(detection_model='cnn', landmarks_model='large')

print('\n\n\n\n ultraLightFD \n\n\n\n')
ultraLightFD = UltraLightOnnxFaceDetector(target_model=320)
fps['ultraLightFD'] = test(ultraLightFD.detect, faceRecognition)
del ultraLightFD


# print('\n\n\n\n yoloV3FaceDetection \n\n\n\n')
# #(pretrained_weights_path,config_file_path,classes_file_path, confidence_threshold=0.9, iou= 0.7):

# from static import yoloV3FaceClassesFilePath, yoloV3FaceConfigFilePath, yoloV3FacePretrainedWeightsPath
# yoloV3FaceDetection = YoloV3FromDarkNet(yoloV3FacePretrainedWeightsPath,yoloV3FaceConfigFilePath, yoloV3FaceClassesFilePath, confidence_threshold=0.99, iou=0.7)
# fps['YoloV3FaceDetection'] = test(yoloV3FaceDetection.detect)
# del yoloV3FaceDetection




# fps['FaceDetection'] = test(faceRecognition.get_face_locations)
# print('\n\n\n\n\n\n',fps)

# atexit.register(faceRecognition.save_known_faces)
