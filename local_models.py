import keras.backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def custom_loss(ps_true, ps_pred):
    """Custom loss function that recreates the WF from the PS and compares them.
    Args:
        ps_true (_type_): The true PS with dim (128, 128, output_turns)
        ps_pred (_type_): The predicted PS with dim (128, 128, output_turns)

    Returns:
        _type_: _description_
    """
    # print(ps_pred.shape, ps_true.shape)

    wf_pred = K.sum(ps_pred, axis=1)
    wf_true = K.sum(ps_true, axis=1)
    # print(wf_pred.shape, wf_true.shape)
    loss = K.mean(K.square(wf_true - wf_pred))
    return loss


class Tomoscope(keras.Model):
    """The model should take as input the waterfall and output Phasespace.
    It should look like an autoencoder, but the output dimension should be different (larger)
    """

    def __init__(self, output_name='tomoscope', input_shape=(128, 128, 1),
                 output_turns = 1, cropping=[[0, 0], [0, 0]],
                 enc_dense_layers=[1024, 256, 64], enc_filters=[8, 16, 32],
                 dec_dense_layers=[256, 1024], dec_filters=[32, 16, 1],
                 enc_kernel_size=3, dec_kernel_size=3,
                 enc_strides=[2, 2], dec_strides=[2,2],
                 enc_activation='relu', dec_activation='relu',
                 final_activation='tanh',
                 enc_pooling=None, dec_pooling=None,
                 enc_pooling_size=[2, 2], dec_pooling_size=[2, 2],
                 enc_pooling_strides=[1, 1], dec_pooling_strides=[1, 1],
                 enc_pooling_padding='valid', dec_pooling_padding='valid',
                 enc_dropout=0.0, dec_dropout=0.0,
                 enc_conv_padding='valid', dec_conv_padding='same',
                 learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=False, batchnorm=False,
                 **kwargs):
        super().__init__()

        self.output_name = output_name

        # The output shape will be (128, 128, output_turns)
        output_shape = (input_shape[0], input_shape[1], output_turns)
        
        # Prepare some of the parameters

        # the kernel_size can be a single int or a list of ints
        if isinstance(enc_kernel_size, int):
            enc_kernel_size = [enc_kernel_size] * len(enc_filters)
        assert len(enc_kernel_size) == len(enc_filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(enc_strides[0], int):
            enc_strides = [enc_strides for _ in enc_filters]
        assert len(enc_strides) == len(enc_filters)

        # the kernel_size can be a single int or a list of ints
        if isinstance(dec_kernel_size, int):
            dec_kernel_size = [dec_kernel_size] * len(dec_filters)
        assert len(dec_kernel_size) == len(dec_filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(dec_strides[0], int):
            dec_strides = [dec_strides for _ in dec_filters]
        assert len(dec_strides) == len(dec_filters)
        
        # Additional checks
        assert dec_conv_padding == 'same', 'The decoder convolution padding must be same'
        assert dec_filters[-1] == output_turns, 'The last decoder filter must be equal to the output_turns'

        # Construct the encoder

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
                use_bias=use_bias, padding=enc_conv_padding,
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
        # middle = x
        # print('middle shape:', middle.shape)

        # The dimension should be equal to (middle_dense_layer, )

        # From this, we want to go to (128, 128, output_turns)

        # Generate the inverse model (encoder) to find the t_shape
        temp = keras.Sequential()
        temp.add(keras.Input(shape=output_shape, name='temp_input'))

        # temp.add(keras.layers.Conv2D(filters=dec_filters[-1], padding='same',
        #                              strides=dec_strides[-1], kernel_size=dec_kernel_size[-1]))
        
        for i in np.arange(len(dec_filters)-1, -1, -1):
            temp.add(keras.layers.Conv2D(filters=dec_filters[i], padding='same',
                                         strides=dec_strides[i], kernel_size=dec_kernel_size[i]))
        # print('Output shape:', output_shape)
        # print(temp.summary())
        t_shape = temp.layers[-1].output_shape[1:]

        del temp

        # print('t_shape:', t_shape)

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

        # extend to needed t_shape and reshape
        x = keras.layers.Dense(units=np.prod(t_shape), activation=dec_activation, name='decoder_expand')(x)
        x = keras.layers.Reshape(target_shape=t_shape, name='decoder_reshape')(x)

        # For every Convolutional layer
        for i, f in enumerate(dec_filters[:]):
            # Add the Convolution
            x = keras.layers.Conv2DTranspose(
                filters=dec_filters[i], kernel_size=dec_kernel_size[i], strides=dec_strides[i],
                activation=dec_activation, name=f'decoder_cnn_{i+1}', padding='same')(x)

        outputs = keras.layers.Activation(activation=final_activation, name='final_activation')(x)

        # print('outputs shape:', outputs.shape)


        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        if loss == 'custom':
            loss = custom_loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    def predict(self, waterfall):
        tomography = self.model(waterfall)
        return tomography

    def load(self, weights_dir, compile=False):
        weights_file = os.path.join(weights_dir, 'tomoscope.h5')
        assert os.path.exists(weights_file), f'Weights file {weights_file} does not exist'
        self.model = keras.models.load_model(weights_file, compile=compile)

    def save(self, weights_file):
        self.model.save(weights_file)

