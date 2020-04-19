import face_recognition
from scipy import spatial
import numpy as np
import os 
import cv2
import random
import string
class FaceRecognition:
    def __init__(self, distance_tolerance=0.6, cosine_tolerance = 0.9, detection_model='cnn', landmarks_model='large'):
        self.distance_tolerance = distance_tolerance
        self.cosine_tolerance = cosine_tolerance
        self.detection_model = detection_model
        self.landmarks_model= landmarks_model
        self.known_face_encodings=[]
        self.known_face_images=[]


    def get_face_locations(self, frame):
        face_locations = face_recognition.face_locations(frame, model=self.detection_model)
        return [(left, top, right, bottom) for (top, right, bottom, left) in face_locations], [0 for i in range(len(face_locations))]

    def get_face_encodings(self, frame, face_locations=None, model="large"):
       return face_recognition.face_encodings(frame, known_face_locations=face_locations, model=model)

    def compare_faces(self, face_encodes_to_compare, tolerance=None):
        if tolerance==None:
            tolerance = self.distance_tolerance
        if(len(self.known_face_encodings)==0):
            return [False]
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encodes_to_compare,tolerance)
        return matches
    
    def get_face_distances(self, face_encodes_to_compare):
        print("face_encodes_to_compare.shape", face_encodes_to_compare.shape)
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encodes_to_compare)
        return face_distances

    def get_face_cosine_similarity(self, face_encodes_to_compare):
        return [1 - spatial.distance.cosine(face_encodes, face_encodes_to_compare) for face_encodes in self.known_face_encodings]

    
    def add_known_face_encodings(self,faces_encodes=None, frame=None, faces_locations=None):
        if (faces_encodes == None and frame==None):
            return None
        
        if faces_encodes != None:
            for encodings in faces_encodes:
                self.known_face_encodings.append(encodings)

        else:
            faces_encodings = self.get_face_encodings(frame,face_locations=faces_locations, model=self.landmarks_model)
            self.known_face_encodings.append(faces_encodings)

        print(len(self.known_face_encodings))

    def randomString(self, stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))
    
    def add_known_face_crop(self,frame, face_location, face_id, save_path):
        x1,y1,x2,y2 = face_location
        os.system("mkdir -p "+save_path+'/'+str(face_id))
        cv2.imwrite(save_path+'/'+str(face_id)+'/'+self.randomString()+'.jpg', frame[y1:y2, x1:x2])

        
        print("add new encodes ", len(self.known_face_encodings))

    def save_known_faces(self):
        np.save('./faceDB/known_faces.npy',self.known_face_encodings)

    
    # def __del__(self):
    #     self.save_known_faces()
        

        






    

