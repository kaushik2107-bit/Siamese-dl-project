import os
import random
import numpy as np
from keras.utils import load_img, img_to_array, save_img
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from tqdm import tqdm
from keras.utils import Sequence
from itertools import combinations
import pandas as pd
from keras.applications.inception_resnet_v2 import preprocess_input

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

def generate_triplets_csv(data_dir, negatives_per_pair, output_csv_path):
    people = os.listdir(data_dir)
    
    count = 0
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["anchor", "positive", "negative"])

        for anchor_person in tqdm(people, desc="Processing anchor persons"):
            anchor_person_path = os.path.join(data_dir, anchor_person)
            anchor_images = os.listdir(anchor_person_path)
            
            if len(anchor_images) < 2: continue
            break_pair = 0
            for anchor_image_name, positive_image_name in combinations(anchor_images, 2):
                for _ in range(negatives_per_pair):
                    negative_person = random.choice([person for person in people if person != anchor_person])
                    negative_person_path = os.path.join(data_dir, negative_person)
                    negative_images = os.listdir(negative_person_path)
    
                    if not negative_images: continue
                
                    negative_image_name = random.choice(negative_images)                
                    writer.writerow([
                        os.path.join(anchor_person_path, anchor_image_name),
                        os.path.join(anchor_person_path, positive_image_name),
                        os.path.join(negative_person_path, negative_image_name)
                    ])
                    count += 1
                break_pair += 1
                if break_pair > 10: break

    print(f"Triplets CSV saved to {output_csv_path}\nTotal Entries {count}")

class BatchGenerator(Sequence):
    def __init__(self, csv_file, batch_size, image_size):
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_data = self.data.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        
        anchors = []
        positives = []
        negatives = []

        for _, row in batch_data.iterrows():
            anchor = self._load_and_preprocess(row["anchor"])
            positive = self._load_and_preprocess(row["positive"])
            negative = self._load_and_preprocess(row["negative"])
            
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return (np.array(anchors), np.array(positives), np.array(negatives)), None

    def on_epoch_end(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def _load_and_preprocess(self, image_path):
        img = load_img(image_path, target_size=self.image_size)
        img = img_to_array(img)
        img = preprocess_input(img)  # Preprocess for VGG16 or similar models
        return img


def generate_triplet_dataset(data_dir, num_triplets_per_person):
    triplets, _ = generate_triplets(data_dir, num_triplets_per_person)
    anchors = triplets[:, 0]
    positives = triplets[:, 1]
    negatives = triplets[:, 2]
    triplet_dataset = tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))
    triplet_dataset = triplet_dataset.map(lambda anchor, positive, negative: ((anchor, positive, negative), ()))
    return triplet_dataset

def triplet_generator(data_dir, batch_size):
    people = os.listdir(data_dir)
    while True:
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(batch_size):
            anchor_person = random.choice(people)
            anchor_person_path = os.path.join(data_dir, anchor_person)
            anchor_images = os.listdir(anchor_person_path)
            
            if len(anchor_images) < 2:
                continue
            anchor_image_name = random.choice(anchor_images)
            positive_image_name = random.choice([img for img in anchor_images if img != anchor_image_name])
            
            anchor_image = load_image(os.path.join(anchor_person_path, anchor_image_name))
            positive_image = load_image(os.path.join(anchor_person_path, positive_image_name))
            
            negative_person = random.choice([person for person in people if person != anchor_person])
            negative_person_path = os.path.join(data_dir, negative_person)
            negative_images = os.listdir(negative_person_path)
            negative_image_name = random.choice(negative_images)
            negative_image = load_image(os.path.join(negative_person_path, negative_image_name))
            
            anchors.append(anchor_image[0])
            positives.append(positive_image[0])
            negatives.append(negative_image[0])
        
        yield (np.array(anchors), np.array(positives), np.array(negatives)), None

def create_triplet_dataset(data_dir, batch_size):
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # Anchor
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # Positive
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # Negative
        ),
        tf.TensorSpec(shape=(), dtype=tf.float32)  # Dummy output (loss doesn't use labels)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: triplet_generator(data_dir, batch_size),
        output_signature=output_signature
    )

    return dataset

def save_random_triplet(triplets, output_dir, num_triplet_to_save=1):
    random_idx = np.random.choice(len(triplets), num_triplet_to_save, replace=False)
    for idx in random_idx:
        anchor, positive, negative = triplets[idx]
        print(anchor.shape, positive.shape, negative.shape)

        save_img(os.path.join(output_dir, f"anchor_{idx}.png"), anchor)
        save_img(os.path.join(output_dir, f"positive_{idx}.png"), positive)
        save_img(os.path.join(output_dir, f"negative_{idx}.png"), negative)

if __name__ == "__main__":
    data_dir = 'eval'
    num_triplets_per_person = 4
    # triplet_dataset = generate_triplet_dataset(data_dir, num_triplets_per_person)
    # triplet_dataset = triplet_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    # print(triplets.shape)
    # save_random_triplet(triplets, "temp", num_triplet_to_save=1)
    # triplet_dataset = create_triplet_dataset(data_dir, 32)
    # triplet_dataset = triplet_dataset.prefetch(tf.data.AUTOTUNE)

    # Debug: Check the structure of a batch
    # for batch in triplet_dataset.take(1):
    #     inputs, labels = batch
    #     anchors, positives, negatives = inputs
    #     print("Anchors shape:", anchors.shape)
    #     print("Positives shape:", positives.shape)
    #     print("Negatives shape:", negatives.shape)
    #     print("Labels shape:", labels.shape)

    csv_file = "triplet_data1_val.csv"
    generate_triplets_csv(data_dir, num_triplets_per_person, csv_file)
    batch_size = 32
    image_size = (224, 224)

    triplet_generator = BatchGenerator(csv_file, batch_size, image_size)
    (X, _) = triplet_generator[0]
    anchors, positives, negatives = X
    print(anchors.shape, positives.shape, negatives.shape)
    print(len(triplet_generator))

