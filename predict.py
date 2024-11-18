import os
import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.preprocessing import normalize
from keras.applications import VGG16

detector = MTCNN()


def load_image(image, target_size=(224, 224)):
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

def detect_and_crop_face(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)

    faces = []
    for detection in detections: #type: ignore
        x, y, width, height = detection['box']
        face = image_rgb[y:y+height, x:x+width]
        faces.append(face)
    return detections, faces

def generate_embedding(model, face_image):
    processed_face = load_image(face_image)
    embedding = model.predict(processed_face)
    print(embedding.shape)
    embedding = normalize(embedding)
    return embedding

def load_embeddings(embeddings_file):
    if os.path.exists(embeddings_file):
        data = np.load(embeddings_file, allow_pickle=True)
        embeddings, labels = data['embeddings'], data['labels']
        embeddings = normalize(embeddings)
        return embeddings, labels
    else:
        return np.empty((0, 1536)), np.array([])

def save_embeddings(embeddings, labels, embeddings_file):
    np.savez(embeddings_file, embeddings=embeddings, labels=labels)

def compare_embeddings(new_embedding, stored_embeddings, stored_labels, threshold=0.5):
    distances = np.linalg.norm(stored_embeddings - new_embedding, axis=1)
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]
    if min_distance < threshold:
        return stored_labels[min_distance_idx], distances[min_distance_idx]
    else: 
        print("Bad Label Predicted: ", stored_labels[min_distance_idx])
        return "Unknown", min_distance

# def rms(distances):
#     return np.sqrt(np.mean(distances ** 2))

# def compare_embeddings(new_embedding, stored_embeddings, stored_labels, threshold=0.5):
#     unique_labels = np.unique(stored_labels)  # Get the unique labels
#     label_distances = {}
    
#     for label in unique_labels:
#         # Find all the stored embeddings corresponding to this label
#         label_indices = np.where(stored_labels == label)[0]
#         label_embeddings = stored_embeddings[label_indices]
        
#         # Calculate the distances between the new embedding and the embeddings of this label
#         distances = np.linalg.norm(label_embeddings - new_embedding, axis=1)
        
#         # Calculate the RMS distance for this label
#         avg_distance = np.mean(distances)
#         label_distances[label] = avg_distance
    
#     # Find the label with the minimum RMS distance
#     closest_label = min(label_distances, key=label_distances.get)
#     min_rms_distance = label_distances[closest_label]
    
#     if min_rms_distance < threshold:
#         return closest_label, min_rms_distance
#     else:
#         print("Bad Label Predicted: ", closest_label)
#         return "Unknown", min_rms_distance
    
def predict(image_path, model, embeddings_file):
    _, faces = detect_and_crop_face(image_path)
    if not faces:
        print("No faces detected.")
        return

    stored_embeddings, stored_labels = load_embeddings(embeddings_file)
    for face in faces:
        embedding = generate_embedding(model, face)
        if len(stored_embeddings) > 0:
            label, distance = compare_embeddings(embedding, stored_embeddings, stored_labels)
            print(f"Most similar person: {label} with distance: {distance}")
        else:
            print("No embeddings available for comparison.")

def add_embedding_to_storage(image_path, label, model, embeddings_file):
    _, faces = detect_and_crop_face(image_path)
    if not faces:
        print("No faces detected.")
        return

    embeddings, labels = load_embeddings(embeddings_file)
    for face in faces:
        embedding = generate_embedding(model, face)
        print(embedding.shape)
        embeddings = np.append(embeddings, embedding, axis=0)
        labels = np.append(labels, label)

    save_embeddings(embeddings, labels, embeddings_file)
    print(f"Embedding for {label} stored successfully.")

def add_entire_folder(path, model, embeddings_file):
    for person_name in os.listdir(path):
        person_folder = os.path.join(path, person_name)

        if os.path.isdir(person_folder):
            print(f"Loading folder for {person_name}...")
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(image_path)
                    add_embedding_to_storage(image_path, person_name, model, embeddings_file)
                    break

def predict_multiple_faces(image_path, model, embeddings_file, output_path, threshold=0.5):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections, faces = detect_and_crop_face(image_path)
    if not faces:
        print("No faces detected.")
        return
    
    stored_embeddings, stored_labels = load_embeddings(embeddings_file)
    
    for i, (face, detection) in enumerate(zip(faces, detections)): # type: ignore
        print(f"Processing face {i+1}...")
        
        embedding = generate_embedding(model, face)
        
        if len(stored_embeddings) > 0:
            label, distance = compare_embeddings(embedding, stored_embeddings, stored_labels, threshold)
            print(f"Face {i+1}: Most similar person: {label} with distance: {distance}")
        else:
            label = "Unknown"
            print(f"Face {i+1}: No embeddings available for comparison.")
        
        x, y, width, height = detection['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)        
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, image)
    print(f"Image saved as {output_path}")

if __name__ == "__main__":
    model_path = 'embedding_model.keras'
    embeddings_file = 'stored_embeddings.npz'

    # model = load_model(model_path)
    model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')

    # embeddings, labels = load_embeddings(embeddings_file)

    add_entire_folder("database", model, embeddings_file)

    image_path = 'test-image3.jpg'
    output_path = "output4.jpg"
    predict_multiple_faces(image_path, model, embeddings_file, output_path, threshold=100)

    # new_face_path = 'new_person.jpg'
    # new_person_label = 'Person_Name'
    # add_embedding_to_storage(new_face_path, new_person_label, model, embeddings_file)
