import tensorflow as tf
from keras.layers import Layer, Input
from keras.models import Model, Sequential
from keras.applications import InceptionResNetV2

class TripletLossLayer(Layer):
    def __init__(self, alpha=0.2, **kwargs):
        super(TripletLossLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + self.alpha, 0.0)
        self.add_loss(tf.reduce_mean(loss))
        return loss

class SiameseNetwork(Model):
    def __init__(self, alpha=0.2, **kwargs):
        super(SiameseNetwork, self).__init__(**kwargs)
        self.alpha = alpha        
        self.model = Sequential()        
        self.model.add(InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg'))        
        self.triplet_loss_layer = TripletLossLayer(alpha=self.alpha)

    def call(self, inputs):
        anchor_input, positive_input, negative_input = inputs
        anchor_embedding = self.model(anchor_input)
        positive_embedding = self.model(positive_input)
        negative_embedding = self.model(negative_input)     

        anchor_embedding = tf.nn.l2_normalize(anchor_embedding, axis=-1)
        positive_embedding = tf.nn.l2_normalize(positive_embedding, axis=-1)
        negative_embedding = tf.nn.l2_normalize(negative_embedding, axis=-1)
        
        return self.triplet_loss_layer([anchor_embedding, positive_embedding, negative_embedding])

    def embedding_model(self):
        input_tensor = Input(shape=(224, 224, 3))
        embedding = self.model(input_tensor)
        return Model(inputs=input_tensor, outputs=embedding)

if __name__ == "__main__":
    anchor_input = Input(name="anchor_input", shape=(224, 224, 3))
    positive_input = Input(name="positive_input", shape=(224, 224, 3))
    negative_input = Input(name="negative_input", shape=(224, 224, 3))

    model = SiameseNetwork()
    embedding_model= model.embedding_model()
    embedding_model.compile(optimizer='adam')
    embedding_model.summary()

    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=model([anchor_input, positive_input, negative_input]))
    siamese_model.compile(optimizer='adam')

    siamese_model.summary()

    from keras.applications import InceptionResNetV2
    inception_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    inception_model.summary()

    from keras.utils import plot_model
    plot_model(inception_model, to_file='inception_model.png', show_shapes=True, show_layer_names=True)
    plot_model(siamese_model, to_file='siamese_model.png', show_shapes=True, show_layer_names=True)
    plot_model(embedding_model, to_file='embedding_model.png', show_shapes=True, show_layer_names=True)