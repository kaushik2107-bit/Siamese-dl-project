import os
from .FaceStorage import FaceStorage
from .FaceDetection import FaceDetector
import dlib
import uuid
from typing import List, Tuple, Dict
import numpy as np

class FaceRecognition:
    keypoints_model_path = "shape_predictor_5_face_landmarks.dat"
    face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    def __init__(self, model_path:str = "./models", data_path:str = "data/facial_data.json", threshold: int = 0.99) -> None:
        keypoints_model_path = os.path.join(model_path, FaceRecognition.keypoints_model_path)
        face_recognition_model_path = os.path.join(model_path, self.face_recognition_model_path)

        if not keypoints_model_path or not os.path.exists(keypoints_model_path): 
            raise Exception("Model File is missing")
        if not face_recognition_model_path or not os.path.exists(face_recognition_model_path):
            raise Exception("Model File is missing")

        self.face_detector = FaceDetector(crop_forehead=True, shrink_ratio=0.2)
        self.threshold = threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recognition_model_path)
        self.datastore = FaceStorage(data_path)

    def register_face(self, image=None, name:str = None, bbox:List[int] = None):
        if name is None:
            raise Exception("No name is provided")
        if image is None:
            raise Exception("Invalid image!!")
        # print(type(image))
        if (len(image.shape) != 3 or image.shape[-1] != 3):
            raise Exception("Invalid image!!")
        
        image = image.copy()
        face_encoding = None
        try:
            if bbox is None:
                bboxes = self.face_detector.detect_faces(image)
                if len(bboxes) == 0:
                    return None
                bbox = bboxes[0]
            face_encoding = self.get_facial_encoding(image, bbox)
            facial_data = {
                "id": str(uuid.uuid4()),
                "encoding": tuple(face_encoding.tolist()),
                "name": name,
            }
            self.save_facial_data(facial_data)
            print(f"Face registered with name {name}")
        except Exception as e:
            raise e
        return facial_data

    def recognize_faces(self, image, threshold: float = 0.6, bboxes: List[List[int]] = None):
        if image is None:
            return Exception("Invalid image")
        image = image.copy()
        if bboxes is None:
            bboxes = self.face_detector.detect_faces(image=image)
        if len(bboxes) == 0:
            return []
        
        all_facial_data = self.datastore.get_all_facial_data()
        matches = []
        for bbox in bboxes:
            face_encoding = self.get_facial_encoding(image, bbox)
            match, min_dist = None, 10000000
            for face_data in all_facial_data:
                dist = self.euclidean_distance(face_encoding, face_data["encoding"])
                if dist <= threshold and dist < min_dist:
                    match = face_data
                    min_dist = dist

            matches.append((bbox, match, min_dist))
        return matches

    def save_facial_data(self, facial_data: Dict = None) -> bool:
        if facial_data is not None:
            self.datastore.add_facial_data(facial_data=facial_data)
            return True
        return False

    def get_registered_faces(self) -> List[Dict]:
        return self.datastore.get_all_facial_data()

    def get_facial_encoding(self, image, bbox:List[int] = None) -> List[float]:
        if bbox is None:
            raise Exception("Missing face")
        
        bbox = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        keypoints = self.keypoints_detector(image, bbox)
        face_encoding = self.get_face_encoding(image, keypoints)
        return face_encoding

    def get_face_encoding(self, image, keypoints: List):
        encoding = self.face_recognizor.compute_face_descriptor(image, keypoints, 1)
        return np.array(encoding)

    def euclidean_distance(self, vector1: Tuple, vector2: Tuple):
        return np.linalg.norm(np.array(vector1) - np.array(vector2))