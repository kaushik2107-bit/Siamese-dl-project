from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from model import SiameseNetwork
from keras.callbacks import History
from generate_dataset import BatchGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def triplet_loss(anchor, positive, negative, margin=0.2):
    positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(positive_distance - negative_distance + margin, 0.0)
    return tf.reduce_mean(loss)

def validation_step(model, generator, steps, margin=0.2):
    total_loss = 0.0
    count = 0
    for step in range(steps):
        (anchors, positives, negatives), _ = generator[count]
        count += 1

        anchor_embeddings = model(anchors, training=False)
        positive_embeddings = model(positives, training=False)
        negative_embeddings = model(negatives, training=False)        
        
        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
        total_loss += loss
        print(step)

    return total_loss / steps

def train_model(model, generator, val_generator, epochs, optimizer, steps_per_epoch):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        batch_count = 0

        for step in range(steps_per_epoch):
            (anchors, positives, negatives), _ = generator[batch_count]

            with tf.GradientTape() as tape:
                anchor_embeddings = model(anchors, training=True)
                positive_embeddings = model(positives, training=True)
                negative_embeddings = model(negatives, training=True)

                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_loss += loss
            batch_count += 1

            if step % 1 == 0:
                print(f"Step {step}/{steps_per_epoch} - Loss: {loss.numpy()}")
            break
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1} finished. Average loss: {avg_loss.numpy()}")

        avg_val_loss = validation_step(model, val_generator, len(val_generator), margin=0.2)
        print(f"Validation Loss: {avg_val_loss.numpy()}")

if __name__ == "__main__":
    data_dir = 'train'
    num_triplets_per_person = 10
    batch_size = 8
    # triplet_dataset = generate_triplet_dataset(data_dir, num_triplets_per_person)
    # triplet_dataset = triplet_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    # triplet_dataset = create_triplet_dataset(data_dir, batch_size)
    # triplet_dataset = triplet_dataset.prefetch(tf.data.AUTOTUNE)

    csv_file = "triplet_data1.csv"
    val_csv_file = "triplet_data1_val.csv"
    image_size = (224, 224)
    triplet_generator = BatchGenerator(csv_file, batch_size, image_size)
    val_triplet_generator = BatchGenerator(val_csv_file, batch_size, image_size)

    anchor_input = Input(name="anchor_input", shape=(224, 224, 3))
    positive_input = Input(name="positive_input", shape=(224, 224, 3))
    negative_input = Input(name="negative_input", shape=(224, 224, 3))

    model = SiameseNetwork()
    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=model([anchor_input, positive_input, negative_input]))
    siamese_model.compile(optimizer='adam')

    siamese_model.summary()

    history = History()

    epochs = 10
    steps_per_epoch = len(triplet_generator)
    embedding_model = model.embedding_model()
    optimizer = Adam(learning_rate=0.0001)
    train_model(embedding_model, triplet_generator, val_triplet_generator, 10, optimizer, steps_per_epoch)

    # siamese_model.fit(triplet_generator, epochs=10, callbacks=[history])

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

