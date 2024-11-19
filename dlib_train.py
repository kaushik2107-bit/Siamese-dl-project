import dlib
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model
import numpy as np
import pandas as pd
from keras.utils import load_img, img_to_array

def load_dlib_resnet_model(weights_path):
    face_rec_model = dlib.face_recognition_model_v1(weights_path)
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")

    def dlib_resnet_embeddings(img_path):
        img = dlib.load_rgb_image(img_path)
        detections = detector(img, 1)  
        if len(detections) == 0:
            return None
            raise ValueError(f"No face detected in image: {img_path}")
        shape = shape_predictor(img, detections[0])
        face_chip = dlib.get_face_chip(img, shape, size=150)
        arr = np.array(face_rec_model.compute_face_descriptor(face_chip)).reshape(1, -1)
        print(arr.shape)
        return arr

    return dlib_resnet_embeddings

def load_triplets(csv_file):
    triplet_data = pd.read_csv(csv_file)
    anchors = triplet_data['anchor'].tolist()
    positives = triplet_data['positive'].tolist()
    negatives = triplet_data['negative'].tolist()
    return anchors, positives, negatives

def preprocess_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    return img_array / 255.0

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + alpha
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

def build_siamese_model(embedding_dim):
    input_anchor = Input(shape=(embedding_dim,))
    input_positive = Input(shape=(embedding_dim,))
    input_negative = Input(shape=(embedding_dim,))

    concatenated = Lambda(lambda x: tf.concat(x, axis=1))([input_anchor, input_positive, input_negative])
    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=concatenated)
    return model

def train_siamese_model_triplet(dlib_model_path, triplet_csv, epochs=10, batch_size=32):
    embedding_fn = load_dlib_resnet_model(dlib_model_path)

    anchors, positives, negatives = load_triplets(triplet_csv)

    def generate_embeddings(image_paths):
        embeddings = []
        for img_path in image_paths:
            embedding = embedding_fn(img_path)
            if embedding is not None:
                embeddings.append(embedding)
        if not embeddings:
            raise ValueError("No valid embeddings found. Check your dataset.")
        return np.vstack(embeddings)

    anchor_embeddings = generate_embeddings(anchors)
    positive_embeddings = generate_embeddings(positives)
    negative_embeddings = generate_embeddings(negatives)

    model = build_siamese_model(128)
    model.compile(optimizer='adam', loss=triplet_loss)
    model.summary()

    model.fit(
        [anchor_embeddings, positive_embeddings, negative_embeddings],
        np.zeros((len(anchor_embeddings),)),
        epochs=epochs,
        batch_size=batch_size
    )

    return model

dlib_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
triplet_csv = "triplet_data2.csv"

model = train_siamese_model_triplet(dlib_model_path, triplet_csv, epochs=10, batch_size=32)
