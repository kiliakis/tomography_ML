import keras.backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np
from local_utils import unnormalize_params, unnormalizeIMG
import os

class Tomoscope(keras.Model):
    """The model should take as input the waterfall and output Phasespace.
    It should look like an autoencoder, but the output dimension should be different (larger)
    """

    def __init__(self, output_name='tomoscope', input_shape=(128, 128, 1),
                 output_turns = 1, cropping=[[0, 0], [0, 0]],
                 enc_dense_layers=[1024, 256, 64], enc_filters=[8, 16, 32],
                 dec_dense_layers=[256, 1024], dec_filters=[32, 16, 8],
                 enc_kernel_size=3, dec_kernel_size=3,
                 enc_strides=[2, 2], dec_strides=[2,2],
                 enc_activation='relu', dec_activation='relu',
                 enc_pooling=None, dec_pooling=None,
                 enc_pooling_size=[2, 2], dec_pooling_size=[2, 2],
                 enc_pooling_strides=[1, 1], dec_pooling_strides=[1, 1],
                 enc_pooling_padding='valid', dec_pooling_padding='valid',
                 enc_dropout=0.0, dec_dropout=0.0,
                 learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=False, batchnorm=False,
                 conv_padding='valid', **kwargs):
        super().__init__()

        self.output_name = output_name
        self.input_shape = input_shape
        # The output shape will be (output_turns, 128, 128, 1)
        self.output_shape = (output_turns,) + input_shape
        
        # Construct the encoder

        # the kernel_size can be a single int or a list of ints
        if isinstance(enc_kernel_size, int):
            enc_kernel_size = [enc_kernel_size] * len(enc_filters)
        assert len(enc_kernel_size) == len(enc_filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(enc_strides[0], int):
            enc_strides = [enc_strides for _ in enc_filters]
        assert len(enc_strides) == len(enc_filters)

        # set the input size
        inputs = keras.Input(shape=input_shape, name='input')
        # crop the edges
        cropped = keras.layers.Cropping2D(cropping=cropping, name='crop')(inputs)
        x = cropped
        # For evey Convolutional layer
        for i, f in enumerate(enc_filters):
            # Add the Convolution
            x = keras.layers.Conv2D(
                filters=f, kernel_size=enc_kernel_size[i], strides=enc_strides[i],
                use_bias=use_bias, padding=conv_padding,
                name=f'encoder_cnn_{i+1}')(x)

            # Apply batchnormalization
            if batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Apply the activation function
            x = keras.activations.get(enc_activation)(x)

            # Optional pooling after the convolution
            if enc_pooling == 'Max':
                x = keras.layers.MaxPooling2D(
                    pool_size=enc_pooling_size, strides=enc_pooling_strides,
                    padding=enc_pooling_padding, name=f'encdoer_maxpooling_{i+1}')(x)
            elif enc_pooling == 'Average':
                x = keras.layers.AveragePooling2D(
                    pool_size=enc_pooling_size, strides=enc_pooling_strides,
                    padding=enc_pooling_padding, name=f'encdoer_averagepooling_{i+1}')(x)

        # Flatten after the convolutions
        x = keras.layers.Flatten(name=f'encoder_flatten')(x)
        # For each optional dense layer
        for i, layer in enumerate(enc_dense_layers):
            # Add the layer
            x = keras.layers.Dense(layer, activation=enc_activation,
                                   name=f'encoder_dense_{i+1}')(x)
            # Add dropout optionally
            if enc_dropout > 0 and enc_dropout < 1:
                x = keras.layers.Dropout(
                    enc_dropout, name=f'encoder_dropout_{i+1}')(x)

        # Add the final layers, one for each output
        outputs = keras.layers.Dense(1, name=output_name)(x)

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model

    def predict(self, waterfall):
        latent = self.model(waterfall)
        return latent

    def load(self, weights_file):
        self.model = keras.models.load_model(weights_file)

    def save(self, weights_file):
        self.model.save(weights_file)