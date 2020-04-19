from mtcnn.mtcnn import MTCNN



class MTCNNDetector:
    '''
    '''

    def __init__(self, threshold = 0.99):
        self.detector = MTCNN()
        self.threshold = threshold

    # box format : 
    # X1, Y1, X2, Y2

    def detect(self, frame):
        boxes = []
        confidences = []
        faces = self.detector.detect_faces(frame)
        for face in faces:
            if (face['confidence']>= self.threshold):
                boxes.append([face['box'][0], face['box'][1], face['box'][0]+face['box'][2], face['box'][1]+face['box'][3]])
                confidences.append(face['confidence'])
        return boxes, confidences