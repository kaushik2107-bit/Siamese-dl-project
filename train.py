from keras.layers import Input
from keras.models import Model
from model import SiameseNetwork
from keras.callbacks import History
from generate_dataset import create_triplet_dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    data_dir = 'train'
    num_triplets_per_person = 10
    batch_size = 32
    # triplet_dataset = generate_triplet_dataset(data_dir, num_triplets_per_person)
    # triplet_dataset = triplet_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    triplet_dataset = create_triplet_dataset(data_dir, batch_size)
    triplet_dataset = triplet_dataset.prefetch(tf.data.AUTOTUNE)

    anchor_input = Input(name="anchor_input", shape=(224, 224, 3))
    positive_input = Input(name="positive_input", shape=(224, 224, 3))
    negative_input = Input(name="negative_input", shape=(224, 224, 3))

    model = SiameseNetwork()
    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=model([anchor_input, positive_input, negative_input]))
    siamese_model.compile(optimizer='adam')

    siamese_model.summary()

    history = History()

    siamese_model.fit(triplet_dataset, epochs=10, callbacks=[history])

    model_save_path = 'siamese_model.keras'
    siamese_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    embedding_model = model.embedding_model()
    embedding_model_save_path = 'embedding_model.keras'
    embedding_model.save(embedding_model_save_path)
    print(f"Embedding model saved to {embedding_model_save_path}")

    # Plot the loss graph
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')  # Save the plot as a PNG file
    plt.close()
    print("Loss plot saved as 'loss_plot.png'")

