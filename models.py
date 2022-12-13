import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import normalize_params
import os


class EncoderSingle():
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, output_name, input_shape, dense_layers, filters,
                 cropping=[[0, 0], [0, 0]], kernel_size=7, strides=[2, 2],
                 activation='relu',
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='valid',
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 metrics=['mae'],
                 **kwargs):

        self.output_name = output_name
        self.inputShape = input_shape
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # set the input size
        inputs = keras.Input(shape=input_shape, name='Input')
        # crop the edges
        cropped = keras.layers.Cropping2D(
            cropping=cropping, name='Crop')(inputs)
        x = cropped
        # For evey Convolutional layer
        for i, f in enumerate(filters):
            # Add the Convolution
            x = keras.layers.Conv2D(
                filters=f, kernel_size=kernel_size[i], strides=strides,
                activation=activation, name=f'{output_name}_CNN_{i+1}')(x)
            # Optional pooling after the convolution
            if pooling == 'Max':
                x = keras.layers.MaxPooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'{output_name}_MaxPooling_{i+1}')(x)
            elif pooling == 'Average':
                x = keras.layers.AveragePooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'{output_name}_AveragePooling_{i+1}')(x)

        # Flatten after the convolutions
        x = keras.layers.Flatten(name=f'{output_name}_Flatten')(x)
        # For each optional dense layer
        for i, layer in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(layer, activation=activation,
                                   name=f'{output_name}_Dense_{i+1}')(x)
            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(
                    dropout, name=f'{output_name}_Dropout_{i+1}')(x)

        # Add the final layers, one for each output
        outputs = keras.layers.Dense(1, name=output_name)(x)

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model


class EncoderMulti():
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, input_shape, dense_layers, filters,
                 cropping=[[0, 0], [0, 0]], kernel_size=3, strides=[2, 2],
                 activation='relu',
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='valid',
                 dropout=0.0, learning_rate=0.002, loss='mse',
                 loss_weights=None,
                 output_names=['phEr', 'enEr', 'bl',
                               'inten', 'Vrf', 'mu', 'VrfSPS'],
                 model_cfg={},
                 **kwargs):

        # assert len(output_names) == dense_layers[-1]
        self.output_names = output_names
        self.inputShape = input_shape
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # The encoder consumes the input and produces the latents features
        # Which are 7: phEr, enEr, bl, inten, Vrf, mu, VrfSPS

        # set the input size
        inputs = keras.Input(shape=input_shape, name='Input')
        # crop the edges
        cropped = keras.layers.Cropping2D(
            cropping=cropping, name='Crop')(inputs)
        self.models = {}
        # Generate multiple models, one for each output
        for var_name in output_names:
            x = cropped
            # For evey Convolutional layer
            for i, f in enumerate(filters):
                # Add the Convolution
                x = keras.layers.Conv2D(
                    filters=f, kernel_size=kernel_size[i], strides=strides,
                    activation=activation, name=f'{var_name}_CNN_{i+1}')(x)
                # Optional pooling after the convolution
                if pooling == 'Max':
                    x = keras.layers.MaxPooling2D(
                        pool_size=pooling_size, strides=pooling_strides,
                        padding=pooling_padding, name=f'{var_name}_MaxPooling_{i+1}')(x)
                elif pooling == 'Average':
                    x = keras.layers.AveragePooling2D(
                        pool_size=pooling_size, strides=pooling_strides,
                        padding=pooling_padding, name=f'{var_name}_AveragePooling_{i+1}')(x)

            # Flatten after the convolutions
            x = keras.layers.Flatten(name=f'{var_name}_Flatten')(x)
            # For each optional dense layer
            for i, layer in enumerate(dense_layers):
                # Add the layer
                x = keras.layers.Dense(layer, activation=activation,
                                       name=f'{var_name}_Dense_{i+1}')(x)
                # Add dropout optionally
                if dropout > 0 and dropout < 1:
                    x = keras.layers.Dropout(
                        dropout, name=f'{var_name}_Dropout_{i+1}')(x)

            outputs = keras.layers.Dense(1, name=var_name)(x)
            # Add the final layers, one for each output
            # outputs.append(keras.layers.Dense(1, name=var_name)(x))
            # Also initialize the optimizer and compile the model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=optimizer, loss=loss)
            self.models[var_name] = model

    @tf.function
    def encode(self, x):
        return self.model(x)


class EncoderFunc():
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, input_shape, dense_layers, filters,
                 cropping=[[0, 0], [0, 0]], kernel_size=3, strides=[2, 2],
                 activation='relu',
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='valid',
                 dropout=0.0, learning_rate=0.002, loss='mse',
                 loss_weights=None,
                 output_names=['phEr', 'enEr', 'bl',
                               'inten', 'Vrf', 'mu', 'VrfSPS'],
                 **kwargs):

        # assert len(output_names) == dense_layers[-1]
        self.output_names = output_names
        self.inputShape = input_shape

        # The encoder consumes the input and produces the latents features
        # Which are 7: phEr, enEr, bl, inten, Vrf, mu, VrfSPS

        # set the input size
        inputs = keras.Input(shape=input_shape)
        # crop the edges
        x = keras.layers.Cropping2D(cropping=cropping, name='Crop')(inputs)
        # For evey Convolutional layer
        for i, f in enumerate(filters):
            # Add the Convolution
            x = keras.layers.Conv2D(
                filters=f, kernel_size=kernel_size, strides=strides,
                activation=activation, name=f'CNN_{i+1}')(x)
            # Optional pooling after the convolution
            if pooling == 'Max':
                x = keras.layers.MaxPooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'MaxPooling_{i+1}')(x)
            elif pooling == 'Average':
                x = keras.layers.AveragePooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'AveragePooling_{i+1}')(x)

        # Flatten after the convolutions
        x = keras.layers.Flatten(name='Flatten')(x)
        # For each optional dense layer
        for i, layer in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(layer, activation=activation,
                                   name=f'Dense_{i+1}')(x)
            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(
                    dropout, name=f'Dropout_{i+1}')(x)

        # Add the final layers, one for each output
        outputs = []
        for var_name in output_names:
            outputs.append(keras.layers.Dense(1, name=var_name)(x))

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)

    @tf.function
    def encode(self, x):
        return self.model(x)


class Encoder(keras.Model):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, input_shape, dense_layers, filters,
                 cropping=[[0, 0], [0, 0]], kernel_size=3, strides=[2, 2],
                 activation='relu',
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='valid',
                 dropout=0.0, learning_rate=0.002, loss='mse',
                 loss_weights=None,
                 **kwargs):
        super(Encoder, self).__init__()
        self.latent_dim = dense_layers[-1]
        self.inputShape = input_shape

        # The encoder consumes the input and produces the latents features
        # Which are 7: phEr, enEr, bl, inten, Vrf, mu, VrfSPS

        # Initialize the model
        self.model = keras.Sequential()
        # set the input size
        self.model.add(keras.layers.InputLayer(input_shape=self.inputShape))
        # crop the edges
        self.model.add(keras.layers.Cropping2D(
            cropping=cropping, name='Crop'))

        # For evey Convolutional layer
        for i, f in enumerate(filters):
            # Add the Convolution
            self.model.add(keras.layers.Conv2D(
                filters=f, kernel_size=kernel_size, strides=strides,
                activation=activation, name=f'CNN_{i+1}'))
            # Optional pooling after the convolution
            if pooling == 'Max':
                self.model.add(keras.layers.MaxPooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'MaxPooling_{i+1}'
                ))
            elif pooling == 'Average':
                self.model.add(keras.layers.AveragePooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'AveragePooling_{i+1}'
                ))
        # t_shape = self.model.layers[-1].output_shape[1:]
        # Flatten after the convolutions
        self.model.add(keras.layers.Flatten(name='Flatten'))
        # For each optional dense layer
        for i, layer in enumerate(dense_layers[:-1]):
            # Add the layer
            self.model.add(keras.layers.Dense(layer, activation=activation,
                                              name=f'Dense_{i+1}'))
            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                self.model.add(keras.layers.Dropout(
                    dropout, name=f'Dropout_{i+1}'))
        # Add the final layer
        self.model.add(keras.layers.Dense(dense_layers[-1], name=f'Output'))

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss,
                           loss_weights=loss_weights)

    @tf.function
    def encode(self, x):
        return self.model(x)


class Decoder(keras.Model):
    def __init__(self, output_shape, dense_layers, filters,
                 kernel_size=3, strides=[2, 2],
                 activation='relu', final_kernel_size=3,
                 dropout=0.0, learning_rate=0.002, loss='mse',
                 **kwargs):
        super(Decoder, self).__init__()
        assert filters[-1] == 1
        assert dense_layers[0] == 8

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
        self.model = keras.Sequential()
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
            strides=1, padding='same'))

        assert self.model.layers[-1].output_shape[1:] == output_shape
        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)

    @tf.function
    def decode(self, z):
        return self.decoder(z)


class encoderDecoderModel():
    var_names = ['phEr', 'enEr', 'bl',
                 'inten', 'Vrf', 'mu',
                 'VrfSPS']

    def __init__(self, enc_weights_dir, dec_weights_dir):
        # The encoder weights are in a single file called decoder.h5
        if 'decoder.h5' in os.listdir(dec_weights_dir):
            self.decoder = keras.models.load_model(
                os.path.join(dec_weights_dir, 'decoder.h5'))
        else:
            print(f'Error, decoder.h5 not found in {dec_weights_dir}')
            exit(-1)
        # The encoder weights are either in a single file called encoder.h5
        if 'encoder.h5' in os.listdir(enc_weights_dir):
            self.encoder = keras.models.load_model(
                os.path.join(enc_weights_dir, 'encoder.h5'))
        else:
            # or in multiple files, one per latent space variable
            models = []
            for var in encoderDecoderModel.var_names:
                fname = f'encoder_{var}.h5'
                if fname in os.listdir(enc_weights_dir):
                    models.append(keras.models.load_model(
                        os.path.join(enc_weights_dir, fname)))
                else:
                    print(f'Error, {fname} not found in {enc_weights_dir}')
                    exit(-1)
            self.encoder = models

    def encode(self, WF):
        if isinstance(self.encoder, list):
            latents = tf.concat([m(WF) for m in self.encoder], axis=1)
        else:
            latents = self.encoder(WF)
        return latents

    def decode(self, latent, turn):
        turn = tf.reshape(turn, [-1, 1])
        extended = tf.concat([latent, turn], axis=1)
        return self.decoder(extended)

    def predictPS(self, WF, turn):
        latent = self.encode(WF)
        PS = self.decode(latent, turn)
        return latent, PS


class extendedCED(keras.Model):

    def __init__(self, latent_dim, additional_latent_dim, input_shape, filters,
                 kernel_size=3, strides=(2, 2),
                 ):
        super(extendedCED, self).__init__()
        self.latent_dim = latent_dim
        self.inputShape = input_shape
        self.additional_latent_dim = additional_latent_dim

        # The encoder consumes the input and produces the latents features
        # Which are: phEr, enEr, bl, inten, Vrf, mu + VrfSPS
        #
        self.encoder = keras.Sequential()
        self.encoder.add(keras.layers.InputLayer(input_shape=self.inputShape))
        for f in filters:
            self.encoder.add(keras.layers.Conv2D(
                filters=f, kernel_size=3, strides=(2, 2), activation='relu'))
        t_shape = self.encoder.layers[-1].output_shape[1:]
        self.encoder.add(keras.layers.Flatten())
        self.encoder.add(keras.layers.Dense(latent_dim))

        self.extender = keras.Sequential()
        self.extender.add(keras.layers.InputLayer(
            input_shape=self.latent_dim + self.additional_latent_dim))

        t_shape = (t_shape[0]+1, t_shape[1]+1, int(t_shape[2]/2))

        self.decoder = keras.Sequential()
        self.decoder.add(keras.layers.InputLayer(
            input_shape=(self.latent_dim + self.additional_latent_dim)))
        self.decoder.add(keras.layers.Dense(
            units=np.prod(t_shape), activation=tf.nn.relu))
        self.decoder.add(keras.layers.Reshape(target_shape=t_shape))
        for f in reversed(filters):
            self.decoder.add(keras.layers.Conv2DTranspose(
                filters=f, kernel_size=3, strides=2, padding='same', activation='relu'))
        self.decoder.add(keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same'))

    @tf.function
    def encode(self, x):
        return self.encoder(x)

    @tf.function
    def decode(self, z):
        return self.decoder(z)

    @tf.function
    def extend(self, encoded_latent_vec, turn_normalized):
        return self.extender(keras.layers.Concatenate()([encoded_latent_vec,
                                                         tf.transpose(keras.layers.Flatten()(turn_normalized))]))

    @tf.function
    def predictPS(self, T_images_input, turn_normalized):
        z = self.encode(T_images_input)
        return self.decode(self.extend(z, turn_normalized)), z


@tf.function
def mse_loss(model, turn_normalized, T_image, PS_image, phErs, enErs, bls, intens, Vrf, mu):
    phErs_norm, enErs_norm, bls_norm, intens_norm, Vrf_norm, mu_norm = normalize_params(
        phErs, enErs, bls, intens, Vrf, mu)
    predicted_beam_logit, latents = model.predictPS(
        T_image, turn_normalized, training=True)
    return keras.metrics.mse(keras.backend.flatten(PS_image),
                             keras.backend.flatten(predicted_beam_logit)),\
        keras.metrics.mse(keras.backend.flatten(latents),
                          keras.backend.flatten(tf.transpose(tf.convert_to_tensor([phErs_norm,
                                                                                   enErs_norm,
                                                                                   bls_norm,
                                                                                   intens_norm,
                                                                                   Vrf_norm,
                                                                                   mu_norm]))))


@tf.function
def mse_loss_encoder(model, T_imgs, phErs, enErs, bls, intens, Vrfs, mus, VrfSPSs):
    phErs, enErs, bls, intens, Vrfs, mus, VrfSPSs = \
        normalize_params(phErs, enErs, bls, intens, Vrfs, mus, VrfSPSs)
    latents = model.encode(T_imgs)
    return keras.metrics.mse(keras.backend.flatten(latents),
                             keras.backend.flatten(
        tf.transpose(tf.convert_to_tensor([phErs,
                                           enErs,
                                           bls,
                                           intens,
                                           Vrfs,
                                           mus,
                                           VrfSPSs]))))


@tf.function
def mse_loss_decoder(model, norm_turns, PS_imgs, phErs, enErs, bls, intens,
                     Vrfs, mus, VrfSPSs):
    norm_pars = tf.transpose(tf.convert_to_tensor(
        normalize_params(phErs, enErs, bls, intens, Vrfs, mus, VrfSPSs)))
    predicted_beam_logit = model.decode(model.extend(norm_pars, norm_turns))
    return keras.metrics.mse(keras.backend.flatten(PS_imgs),
                             keras.backend.flatten(predicted_beam_logit))
