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
        # The output shape will be (128, 128, output_turns)
        self.output_shape = (input_shape[0], input_shape[1], output_turns)
        
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

        # Middle layer
        middle = x
        
        # Now generate the decoder
        # For each optional dense layer
        for i, layer in enumerate(dec_dense_layers):
            # Add the layer
            x = keras.layers.Dense(layer, activation=dec_activation,
                                   name=f'decoder_dense_{i+1}')(x)
            # Add dropout optionally
            if dec_dropout > 0 and dec_dropout < 1:
                x = keras.layers.Dropout(
                    dec_dropout, name=f'decoder_dropout_{i+1}')(x)
        

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

        
class Decoder(keras.Model):
    def __init__(self, output_shape=(128, 128, 1), dense_layers=[8, 64, 1024],
                 filters=[32, 16, 8, 1],
                 kernel_size=3, strides=[2, 2],
                 activation='relu', final_kernel_size=3,
                 final_activation='linear',
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 **kwargs):
        super().__init__()
        assert filters[-1] == 1
        # assert dense_layers[0] == 8

        # I calculate the dimension if I was moving:
        # output_shape --> filter[-1] --> filter[-2] .. filter[0]

        # Generate the inverse model (encoder) to find the t_shape
        temp = keras.Sequential()
        temp.add(keras.Input(shape=output_shape))
        temp.add(keras.layers.Conv2D(filters=filters[-1], padding='same',
                                     strides=1, kernel_size=final_kernel_size))

        for filter in (filters[::-1])[1:]:
            temp.add(keras.layers.Conv2D(filters=filter, padding='same',
                                         strides=strides, kernel_size=kernel_size))

        t_shape = temp.layers[-1].output_shape[1:]
        # print(temp.summary())
        del temp
        # t_shape = (t_shape[0]+1, t_shape[1]+1, int(t_shape[2]/2))

        # Initialize the model
        self.model = keras.Sequential(name='Decoder')
        # set the input size
        self.model.add(keras.Input(shape=dense_layers[0], name='Input'))

        # For each optional dense layer
        for i, layer in enumerate(dense_layers[1:]):
            # Add the layer
            self.model.add(keras.layers.Dense(layer, activation=activation,
                                              name=f'Dense_{i+1}'))
            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                self.model.add(keras.layers.Dropout(
                    dropout, name=f'Dropout_{i+1}'))

        # extend to needed t_shape and reshape
        self.model.add(keras.layers.Dense(
            units=np.prod(t_shape), activation='relu', name='Expand'))
        self.model.add(keras.layers.Reshape(
            target_shape=t_shape, name='Reshape'))

        # For evey Convolutional layer
        for i, f in enumerate(filters[:-1]):
            # Add the Convolution
            self.model.add(keras.layers.Conv2DTranspose(
                filters=f, kernel_size=kernel_size, strides=strides,
                activation=activation, name=f'CNN_{i+1}', padding='same'))

        # Final output convolution
        self.model.add(keras.layers.Conv2DTranspose(
            filters=filters[-1], kernel_size=final_kernel_size,
            strides=1, padding='same', name=f'CNN_final'))

        self.model.add(keras.layers.Activation(
            activation=final_activation, name='final_activation'))

        assert self.model.layers[-1].output_shape[1:] == output_shape
        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)

    def load(self, weights_dir):
        # The encoder weights are in a single file called decoder.h5
        if 'decoder.h5' in os.listdir(weights_dir):
            self.model = keras.models.load_model(
                os.path.join(weights_dir, 'decoder.h5'),
                compile=False)
            optimizer = keras.optimizers.Adam(learning_rate=1e-3)
            self.model.compile(optimizer=optimizer, loss='mse')
        else:
            raise FileNotFoundError(
                f'File decoder.h5 not found in {weights_dir}')

    def predict(self, latent, turn, unnormalize=False):
        turn = tf.reshape(turn, [-1, 1])
        extended = tf.concat([turn, latent], axis=1)
        # check input size and drop dimension if needed
        PS = self.model.predict(extended)
        if unnormalize:
            PS = unnormalizeIMG(PS)
        return PS

    def save(self, model_path):
        file_path = os.path.join(model_path, 'decoder.h5')
        self.model.save(file_path)

    def summary(self):
        self.model.summary()

