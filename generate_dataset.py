import os
import random
import numpy as np
from keras.utils import load_img, img_to_array, save_img
import matplotlib.pyplot as plt
import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def generate_triplets(data_dir, num_triplets_per_person):
    people = os.listdir(data_dir)
    triplets = []
    anchor_names = []
    
    for anchor_person in people:
        anchor_person_path = os.path.join(data_dir, anchor_person)
        anchor_images = os.listdir(anchor_person_path)
        
        if len(anchor_images) < 2:
            continue
        
        for _ in range(num_triplets_per_person):
            anchor_image_name = random.choice(anchor_images)
            positive_image_name = random.choice([img for img in anchor_images if img != anchor_image_name])

            anchor_image = load_image(os.path.join(anchor_person_path, anchor_image_name))
            positive_image = load_image(os.path.join(anchor_person_path, positive_image_name))
            
            negative_person = random.choice([person for person in people if person != anchor_person])
            negative_person_path = os.path.join(data_dir, negative_person)
            negative_images = os.listdir(negative_person_path)
            
            negative_image_name = random.choice(negative_images)
            negative_image = load_image(os.path.join(negative_person_path, negative_image_name))
            
            triplet = np.vstack([anchor_image, positive_image, negative_image])
            triplets.append(triplet)
            anchor_names.append(anchor_person)
    
    triplets = np.array(triplets)
    anchor_names = np.array(anchor_names)
    return triplets, anchor_names

def generate_triplet_dataset(data_dir, num_triplets_per_person):
    triplets, _ = generate_triplets(data_dir, num_triplets_per_person)
    anchors = triplets[:, 0]
    positives = triplets[:, 1]
    negatives = triplets[:, 2]
    triplet_dataset = tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))
    triplet_dataset = triplet_dataset.map(lambda anchor, positive, negative: ((anchor, positive, negative), ()))
    return triplet_dataset

def save_random_triplet(triplets, output_dir, num_triplet_to_save=1):
    random_idx = np.random.choice(len(triplets), num_triplet_to_save, replace=False)
    for idx in random_idx:
        anchor, positive, negative = triplets[idx]
        print(anchor.shape, positive.shape, negative.shape)

        save_img(os.path.join(output_dir, f"anchor_{idx}.png"), anchor)
        save_img(os.path.join(output_dir, f"positive_{idx}.png"), positive)
        save_img(os.path.join(output_dir, f"negative_{idx}.png"), negative)

if __name__ == "__main__":
    data_dir = 'train'
    num_triplets_per_person = 50
    triplet_dataset = generate_triplet_dataset(data_dir, num_triplets_per_person)
    triplet_dataset = triplet_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    # print(triplets.shape)
    # save_random_triplet(triplets, "temp", num_triplet_to_save=1)
