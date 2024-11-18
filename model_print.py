from keras import layers, models, Input
from keras.initializers import he_normal
from keras.backend import backend as K
import tensorflow as tf

def conv_block(x, filters, kernel_size=1, strides=1, use_bias=False):
    """
    A basic convolutional block with a convolution, batch normalization, and ReLU activation.
    """
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=use_bias, 
                      kernel_initializer=he_normal())(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def identity_block(x, filters):
    """
    The identity block is used in ResNet. It has skip connections.
    """
    shortcut = x
    
    # First convolution
    x = conv_block(x, filters, kernel_size=1, strides=1)
    
    # Second convolution (same as first but with different filters)
    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                      kernel_initializer=he_normal())(x)
    x = layers.BatchNormalization()(x)
    
    # Adding the shortcut (skip connection)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def conv2d_block(x, filters, kernel_size=1, strides=2):
    """
    A convolutional block with a skip connection that reduces the spatial dimensions.
    """
    shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False,
                             kernel_initializer=he_normal())(x)
    shortcut = layers.BatchNormalization()(shortcut)
    
    # Regular convolution block
    x = conv_block(x, filters, kernel_size=kernel_size, strides=strides)
    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                      kernel_initializer=he_normal())(x)
    x = layers.BatchNormalization()(x)
    
    # Adding the shortcut (skip connection)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def resnet34(input_shape=(224, 224, 3), embedding_dim=128):
    """
    Manually create a ResNet-34 model with a 128-dimensional embedding output (first 29 layers).
    """
    inputs = Input(shape=input_shape)
    
    # Initial Conv Block with kernel size 3 (instead of 7)
    x = conv_block(inputs, 64, kernel_size=3, strides=2)
    
    # Max pooling
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    # Stage 1: 3 blocks with 64 filters
    for _ in range(3):
        x = identity_block(x, 64)
    
    # Stage 2: 4 blocks with 128 filters
    x = conv2d_block(x, 128, kernel_size=1, strides=2)  # Half the kernel size from 3 to 1
    for _ in range(3):
        x = identity_block(x, 128)
    
    # Stage 3: 6 blocks with 256 filters
    x = conv2d_block(x, 256, kernel_size=1, strides=2)  # Half the kernel size from 3 to 1
    for _ in range(5):
        x = identity_block(x, 256)
    
    # Stage 4: 3 blocks with 512 filters
    x = conv2d_block(x, 512, kernel_size=1, strides=2)  # Half the kernel size from 3 to 1
    for _ in range(2):
        x = identity_block(x, 512)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layer for embeddings (output 128-dimensional vector)
    x = layers.Dense(embedding_dim, activation=None)(x)
    
    # Normalize the embeddings (optional for metric learning)
    x = layers.Lambda(lambda t: t / tf.sqrt(tf.reduce_sum(t**2, axis=1, keepdims=True)))(x)
    
    # Create the model up to the 29th layer
    model = models.Model(inputs, x, name="ResNet34_Embedding_29_Layers")
    return model

# Create the model
resnet34_model = resnet34()

# Print model summary
resnet34_model.summary()
from keras.utils import plot_model
plot_model(resnet34_model, to_file='inception_model.png', show_shapes=True, show_layer_names=True)
