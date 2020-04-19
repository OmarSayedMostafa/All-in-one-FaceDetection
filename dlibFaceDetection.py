import dlib
import cv2
from static import dlib_mmod_model_Path
#----------------------------------------------------------------------------------

class dlibFaceDetector:
    '''
    #Contains 2 Method:
        1. Dlib MMOD DNN : Maximum-Margin Object Detector ( MMOD ) with CNN based features.
            # Pros
                1. Works for different face orientations
                2. Robust to occlusion
                3. Works very fast on GPU
                4. Very easy training process
            # Cons
                1. Very slow on CPU
                2. Does not detect small faces as it is trained for minimum face size of 80×80. Thus, you need to make sure that the face size should be more than that in your application. You can however, train your own face detector for smaller sized faces.
                3. The bounding box is even smaller than the HoG detector.

        #==============================================================================================================================
        
        2. Dlib HOG Detector: Based on HoG features and SVM:
            # Pros
                1. Fastest method on CPU
                2. Works very well for frontal and slightly non-frontal faces
                3. Light-weight model as compared to the other three.
                4. Works under small occlusion
                
                Basically, this method works under most cases except a few as discussed below.

            # Cons
                1. The major drawback is that it does not detect small faces as it is trained for minimum face size of 80×80. Thus, you need to make sure that the face size should be more than that in your application. You can however, train your own face detector for smaller sized faces.
                2. The bounding box often excludes part of forehead and even part of chin sometimes.
                3. Does not work very well under substantial occlusion
                4. Does not work for side face and extreme non-frontal faces, like looking down or up.
        
        #==============================================================================================================================

    source : https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
    code on github: https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison
    '''
    def __init__(self, target_method='MMOD', threshold=0.99, scale_factor=1.0):
        self.threshold = threshold
        self.scale_factor = scale_factor

        target_method = target_method.lower()
        if target_method == 'mmod':
            print('========================')
            print('Using MMOD Model')
            print('========================')
            self.detector = dlib.cnn_face_detection_model_v1(dlib_mmod_model_Path)
        
        else:
            print('========================')
            print('Using HOG Method')
            print('========================')
            self.detector = dlib.get_frontal_face_detector()



    def detect(self, frame):
        frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faceRects = self.detector(frame, 0)
        bboxes = []
        for faceRect in faceRects:
            try:
                bboxes.append([faceRect.rect.left()*(1.0/self.scale_factor), faceRect.rect.top()*(1.0/self.scale_factor), faceRect.rect.right()*(1.0/self.scale_factor), faceRect.rect.bottom()*(1.0/self.scale_factor) ])
            except:
                print("boxes",bboxes)
                print("faceRect",faceRect)
        return bboxes, None
