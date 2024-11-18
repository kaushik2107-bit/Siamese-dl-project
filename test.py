import os
import shutil
import random
from src.FaceRecognition import FaceRecognition
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import glob
import numpy as np
import cv2

def load_image(image_path, mode:str = "rgb"):
    try:
        img = cv2.imread(image_path)
        if mode == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise e

def test_generation(train_dir, database_test_dir, test_model_dir):
    os.makedirs(database_test_dir, exist_ok=True)
    os.makedirs(test_model_dir, exist_ok=True)

    for person in os.listdir(train_dir):
        person_dir = os.path.join(train_dir, person)
        if os.path.isdir(person_dir):
            images = [img for img in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, img))]

            if len(images) < 4:
                print(f"Skipping {person} as they have less than 4 images.")
                continue

            selected_images = random.sample(images, 4)
            db_test_image = selected_images[0]
            db_test_image_src = os.path.join(person_dir, db_test_image)
            db_test_image_dst = os.path.join(database_test_dir, f"{person}.jpg")
            shutil.copy(db_test_image_src, db_test_image_dst)

            for i, test_image in enumerate(selected_images[1:], start=1):
                test_image_src = os.path.join(person_dir, test_image)
                test_image_dst = os.path.join(test_model_dir, f"{person}_{i}.jpg")
                shutil.copy(test_image_src, test_image_dst)

            print(f"Processed {person}: 1 image to database_test, 3 images to test_model.")


def evaluate_recognition(test_model_dir, face_recognizer, threshold=0.6):
    results = {}
    all_labels = []
    predictions = []

    count = 0
    for image_name in os.listdir(test_model_dir):
        image_path = os.path.join(test_model_dir, image_name)
        img = load_image(image_path)
        matches = face_recognizer.recognize_faces(image=img)
        expected_label = image_name.split("_")[0]
        all_labels.append(expected_label)  

        # print(matches)
        if matches:
            best_match = min(matches, key=lambda x: x[-1])  
            predicted_label = best_match[1]["name"] if best_match[-1] <= threshold else "Unknown"
        else:
            predicted_label = "Unknown"

        predictions.append(predicted_label)  
        matched = predicted_label == expected_label
        results[image_name] = matched
        print(f"Image: {image_name}, Expected: {expected_label}, Matched: {predicted_label}")

        count += 1
        # if count > 10: break
    return results, all_labels, predictions

def compute_confusion_matrix(all_labels, predictions):
    unique_labels = sorted(set(all_labels + predictions))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    y_true = [label_to_index[label] for label in all_labels]
    y_pred = [label_to_index[label] for label in predictions]

    confusion_mat = confusion_matrix(y_true, y_pred, labels=range(len(unique_labels)))

    print("\nConfusion Matrix:")
    print(confusion_mat)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=unique_labels))

    TP = np.diag(confusion_mat).sum()
    FP = confusion_mat.sum(axis=0) - np.diag(confusion_mat)
    FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
    TN = confusion_mat.sum() - (FP + FN + TP)

    FP = FP.sum()
    FN = FN.sum()
    TN = TN.sum()

    print("\nOverall Metrics:")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Negatives (TN): {TN}")

    # Compute overall precision, recall, and F1-score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")


    return confusion_mat

if __name__ == "__main__":
    json_path = "data/facial_data_test.json"
    # test_generation("train", "database_test", "model_test")

    face_recognizer = FaceRecognition(data_path=json_path)

    # for file in glob.glob("database_test/*"):
    #     person_name = os.path.splitext(os.path.basename(file))[0]
    #     img = load_image(file)
    #     face_recognizer.register_face(image=img, name=person_name)

    _, all_labels, predictions = evaluate_recognition("model_test", face_recognizer)
    confusion_mat = compute_confusion_matrix(all_labels, predictions)

