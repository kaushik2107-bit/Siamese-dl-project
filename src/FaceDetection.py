import sys 
from typing import List
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self, crop_forehead: bool = True, shrink_ratio: int = 0.1) -> None:
        try:
            self.detector = MTCNN()
            self.forehead = crop_forehead
            self.shrink_ratio = shrink_ratio
        except Exception as e:
            raise e
        
    def detect_faces(self, image, conf_threshold: float = 0.7) -> List[List[int]]:
        if image is None:
            raise Exception("Invalid image!!")
        if (len(image.shape) != 3 or image.shape[-1] != 3):
            raise Exception("Invalid image!!")
        
        detections = self.detector.detect_faces(image)

        bboxes = []
        for detection in detections:
            conf = detection["confidence"]
            if conf >= conf_threshold:
                x, y, w, h = detection["box"]
                x1, y1, x2, y2 = x, y, x+w, y+h
                if self.forehead:
                    y1 = y1 + int(h * self.shrink_ratio)
                bboxes.append([x1, y1, x2, y2])

        return bboxes
        