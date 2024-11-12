from keras.layers import Input
from keras.models import Model
from model import SiameseNetwork
from keras.callbacks import History
from generate_dataset import generate_triplets
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    data_dir = 'train'
    num_triplets_per_person = 10
    triplets, anchor_names = generate_triplets(data_dir, num_triplets_per_person)
    print(triplets.shape, anchor_names.shape)

    anchor_input = Input(name="anchor_input", shape=(224, 224, 3))
    positive_input = Input(name="positive_input", shape=(224, 224, 3))
    negative_input = Input(name="negative_input", shape=(224, 224, 3))

    model = SiameseNetwork()
    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=model([anchor_input, positive_input, negative_input]))
    siamese_model.compile(optimizer='adam')

    siamese_model.summary()

    history = History()

    print(triplets[:, 0].shape)
    print(triplets[:, 1].shape)
    print(triplets[:, 2].shape)
    siamese_model.fit([triplets[:, 0], triplets[:, 1], triplets[:, 2]], 
                      None, 
                      epochs=10, 
                      batch_size=32)

    model_save_path = 'siamese_model.keras'
    siamese_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot the loss graph
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')  # Save the plot as a PNG file
    plt.close()
    print("Loss plot saved as 'loss_plot.png'")

