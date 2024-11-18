import glob
import cv2
import os
from src.FaceRecognition import FaceRecognition

def load_image(image_path, mode:str = "rgb"):
    try:
        img = cv2.imread(image_path)
        if mode == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise e
    
face_recognizer = FaceRecognition()
# for file in glob.glob("database/*"):
#     person_name = os.path.splitext(os.path.basename(file))[0]
#     img = load_image(file)
#     face_recognizer.register_face(image=img, name=person_name)

img = load_image("test/test2.jpg")
matches = face_recognizer.recognize_faces(image=img)
# print(matches)

for bbox, match, min_dist in matches:
    if match is None: continue  
    x1, y1, x2, y2 = bbox
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)    
    # print(match)
    label = f"{match['name']}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

output_path = "results/test2-output.jpg"
cv2.imwrite(output_path, img)
print(f"Annotated image saved at {output_path}")
