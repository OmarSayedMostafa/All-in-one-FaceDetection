import cv2
from static import TFconfigFile, TFmodelFile, CAFFEconfigFile, CAFFEmodelFile



class opencvDNNFaceDetector:
    '''
    Single-Shot-Multibox detector and uses ResNet-10 Architecture as backbone.
    OpenCV provides 2 models for this face detector:
        1. FloatPoint16 version of the original caffe implementation ( 5.4 MB )
        2. 8 bit Quantized version using Tensorflow ( 2.7 MB )


    # Pros

        1. Runs at real-time on CPU
        2. Works for different face orientations – up, down, left, right, side-face etc.
        3. Works even under substantial occlusion
        4. Detects faces across various scales ( detects big as well as tiny faces )
    
    source : https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
    code on github : https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison
    '''

    def __init__(self, target_model='TF', threshold=0.99):
        
        self.threshold = threshold

        if target_model=='TF' or target_model == 'tf':
            print('===========================================================')
            print('Using 8 bit Quantized version using Tensorflow ( 2.7 MB )')
            print('===========================================================')
            # 8 bit Quantized version using Tensorflow ( 2.7 MB )
            self.net = cv2.dnn.readNetFromTensorflow(TFmodelFile, TFconfigFile)
        else:
            print('===========================================================================')
            print('Using FloatPoint16 version of the original caffe implementation ( 5.4 MB )')
            print('===========================================================================')

            # FP16 version of the original caffe implementation ( 5.4 MB )
            self.net = cv2.dnn.readNetFromCaffe(CAFFEconfigFile, CAFFEmodelFile)

    
    # box format : 
    # X1, Y1, X2, Y2
    def detect(self, frame):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # prepare image tensor for prediction by normalise and resize and scaling
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
        return bboxes



