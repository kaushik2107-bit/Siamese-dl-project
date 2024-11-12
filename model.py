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
        return self.triplet_loss_layer([anchor_embedding, positive_embedding, negative_embedding])

if __name__ == "__main__":
    anchor_input = Input(name="anchor_input", shape=(224, 224, 3))
    positive_input = Input(name="positive_input", shape=(224, 224, 3))
    negative_input = Input(name="negative_input", shape=(224, 224, 3))

    model = SiameseNetwork()
    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=model([anchor_input, positive_input, negative_input]))
    siamese_model.compile(optimizer='adam')

    siamese_model.summary()
