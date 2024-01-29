import keras.backend as K
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os

from utils import unnormalize_params, unnormalizeIMG




def calculate_padding(input_shape, target_shape):
    # Calculate the padding needed for the first two dimensions
    padding_first_dim = (target_shape[0] - input_shape[0]) // 2
    mod_first_dim = (target_shape[0] - input_shape[0]) % 2
    padding_second_dim = (target_shape[1] - input_shape[1]) // 2
    mod_second_dim = (target_shape[1] - input_shape[1]) % 2

    # If the padding doesn't divide evenly, add the extra padding to one side
    pad_first_dim_left = padding_first_dim + mod_first_dim
    pad_second_dim_left = padding_second_dim + mod_second_dim

    # Create the padding configuration for np.pad
    padding_config = (
        # Padding for the first dimension
        (pad_first_dim_left, padding_first_dim),
        # Padding for the second dimension
        (pad_second_dim_left, padding_second_dim),
        # (0, 0)  # Padding for the third dimension
    )

    return padding_config


class BaseModel(keras.Model):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = None

    def predict(self, x):
        return self.model(x)

    def load(self, weights_file):
        self.model = keras.models.load_model(weights_file)

    def save(self, weights_file):
        self.model.save(weights_file)


# model definition
class AutoEncoderSkipAhead(BaseModel):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, output_name='autoencoder',
                 input_shape=(128, 128, 1), dense_layers=[7],
                 decoder_dense_layers=[],
                 cropping=[[0, 0], [0, 0]],
                 filters=[8, 16, 32],  kernel_size=3, conv_padding='same',
                 strides=[2, 2], activation='relu',
                 final_activation='linear', final_kernel_size=3,
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=True, conv_batchnorm=False,
                 dense_batchnorm=False, **kwargs):
        super().__init__()

        self.output_name = output_name
        self.inputShape = input_shape
        # the kernel_size can be a single int or a list of ints
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(strides[0], int):
            strides = [strides for _ in filters]
        assert len(strides) == len(filters)

        # set the input size
        inputs = keras.Input(shape=input_shape, name='Input')

        # crop the edges
        cropped = keras.layers.Cropping2D(cropping=cropping, name='Crop')(inputs)
        x = cropped

        skip_layers = []

        # For evey Convolutional layer
        for i, f in enumerate(filters):
            # Add the Convolution
            x = keras.layers.Conv2D(
                filters=f, kernel_size=kernel_size[i], strides=strides[i],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_{i+1}')(x)

            # Apply batchnormalization
            if conv_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Apply the activation function
            x = keras.activations.get(activation)(x)

            # append to skip layers
            skip_layers.append(x)

        # we have reached the latent space
        last_shape = x.shape[1:]
        x = keras.layers.Flatten(name='Flatten')(x)
        flat_shape = x.shape[1:]
        # Now we add the dense layers
        for i, units in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'encoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # a dummy layer just to name it latent space
        x = keras.layers.Lambda(lambda x: x, name='LatentSpace')(x)
        self.encoder = keras.Model(inputs=inputs, outputs=x, name='encoder')

        # Now we add the decoder dense layers
        for i, units in enumerate(decoder_dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'decoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # Now reshape back to last_shape
        x = keras.layers.Dense(units=np.prod(flat_shape), activation=activation,
                               name='decoder_dense_final')(x)

        x = keras.layers.Reshape(target_shape=last_shape, name='Reshape')(x)
        # Now with transpose convolutions we go back to the original size

        for i, f in enumerate(filters[::-1]):
            x = keras.layers.Concatenate(name=f'Concatenate_{i+1}')([x, skip_layers[-i-1]])
            x = keras.layers.Conv2DTranspose(
                filters=f, kernel_size=kernel_size[-i-1],
                strides=strides[-i-1],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_Transpose_{i+1}')(x)

        # final convolution to get the right number of channels
        x = keras.layers.Conv2DTranspose(filters=1, kernel_size=final_kernel_size,
                                         strides=1, use_bias=use_bias, padding='same',
                                         name=f'CNN_Transpose_Final')(x)

        x = keras.layers.Activation(activation=final_activation,
                                    name='final_activation')(x)
        before_padding = x
        # Add zero padding
        padding = calculate_padding(
            input_shape=before_padding.shape[1:], target_shape=input_shape)
        outputs = keras.layers.ZeroPadding2D(
            padding=padding, name='Padding')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=output_name)

        # assert model.layers[-1].output_shape[1:] == input_shape

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model

# model definition


# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VariationalAutoEncoder(BaseModel):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, output_name='vae',
                 input_shape=(128, 128, 1), dense_layers=[7],
                 decoder_dense_layers=[], latent_dim=2,
                 cropping=[[0, 0], [0, 0]],
                 filters=[8, 16, 32],  kernel_size=3, conv_padding='same',
                 strides=[2, 2], activation='relu',
                 final_activation='sigmoid', final_kernel_size=3,
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=True, conv_batchnorm=False,
                 dense_batchnorm=False, **kwargs):
        super().__init__()

        self.output_name = output_name
        self.inputShape = input_shape
        # the kernel_size can be a single int or a list of ints
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(strides[0], int):
            strides = [strides for _ in filters]
        assert len(strides) == len(filters)

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
                filters=f, kernel_size=kernel_size[i], strides=strides[i],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_{i+1}')(x)

            # Apply batchnormalization
            if conv_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Apply the activation function
            x = keras.activations.get(activation)(x)

        # we have reached the latent space
        last_shape = x.shape[1:]
        x = keras.layers.Flatten(name='Flatten')(x)
        flat_shape = x.shape[1:]
        # Now we add the dense layers
        for i, units in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'encoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        z_mean = keras.layers.Dense(units=latent_dim)(x)
        z_log_var = keras.layers.Dense(units=latent_dim)(x)
        
        z = keras.layers.Lambda(sampling, name='LatentSpace')([z_mean, z_log_var])

        # a dummy layer just to name it latent space
        # x = keras.layers.Lambda(lambda x: x, name='LatentSpace')(x)
        # self.encoder = keras.Model(inputs=inputs, outputs=z, name='encoder')

        # Decoder network

        decoder_inputs = keras.layers.Input(shape=(latent_dim,), name='decoder_input')
        # Now we add the decoder dense layers
        for i, units in enumerate(decoder_dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'decoder_dense_{i+1}')(decoder_inputs)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # Now reshape back to last_shape
        x = keras.layers.Dense(units=np.prod(flat_shape), activation=activation,
                               name='decoder_dense_final')(x)

        x = keras.layers.Reshape(target_shape=last_shape, name='Reshape')(x)
        # Now with transpose convolutions we go back to the original size

        for i, f in enumerate(filters[::-1]):
            x = keras.layers.Conv2DTranspose(
                filters=f, kernel_size=kernel_size[-i-1],
                strides=strides[-i-1],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_Transpose_{i+1}')(x)

        # final convolution to get the right number of channels
        x = keras.layers.Conv2DTranspose(filters=1, kernel_size=final_kernel_size,
                                         strides=1, use_bias=use_bias, padding='same',
                                         name=f'CNN_Transpose_Final')(x)

        x = keras.layers.Activation(activation=final_activation,
                                    name='final_activation')(x)

        before_padding = x
        # Add zero padding
        padding = calculate_padding(
            input_shape=before_padding.shape[1:], target_shape=input_shape)
        outputs = keras.layers.ZeroPadding2D(padding=padding, name='Padding')(x)

        self.encoder = keras.Model(inputs=inputs, outputs=[
                                   z_mean, z_log_var, z], name='encoder')
        self.decoder = keras.Model(inputs=decoder_inputs, outputs=outputs, name='decoder')

        outputs = self.decoder(self.encoder(inputs)[2])
        model = keras.Model(inputs=inputs, outputs=outputs, name=output_name)

        # Loss function
        # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, outputs), axis=(1, 2)))
        reconstruction_loss = tf.reduce_mean(tf.abs(inputs-outputs))

        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        vae_loss = reconstruction_loss + kl_loss

        # model = keras.Model(inputs=inputs, outputs=outputs, name=output_name)

        # assert model.layers[-1].output_shape[1:] == input_shape

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.add_loss(vae_loss)
        model.compile(optimizer=optimizer, metrics=metrics)

        self.model = model


# model definition
class AutoEncoderEfficientNet(BaseModel):
    def __init__(self, output_name='autoencoder',
                 input_shape=(128, 128, 1), 
                 cropping=[[0, 0], [0, 0]],
                 dense_layers=[256], dense_batchnorm=False,
                 reshape_shape=(13, 13, 1),
                 filters=[8, 16, 32], kernel_size=3, conv_padding='same',
                 strides=[2, 2], activation='relu',
                 final_activation='linear', final_kernel_size=3,
                 dropout=0.0, learning_rate=0.001, loss='mae',
                 metrics=[], use_bias=True, **kwargs):
        super().__init__()

        self.output_name = output_name
        self.inputShape = input_shape

        
        # the kernel_size can be a single int or a list of ints
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(strides[0], int):
            strides = [strides for _ in filters]
        assert len(strides) == len(filters)

        # set the input size
        inputs = keras.Input(shape=input_shape, name='Input')

        # this is the autoencoder case
        # crop the edges
        cropped = keras.layers.Cropping2D(
            cropping=cropping, name='Crop')(inputs)
        x = cropped


        efficientnet = tf.keras.Sequential([
            keras.applications.EfficientNetV2B0(include_top=False,
                                        pooling='avg',
                                        weights=None,
                                        classifier_activation='relu',
                                        include_preprocessing=False,
                                        input_shape=x.shape[1:])])
        x = efficientnet(x)

        x = keras.layers.Flatten(name='Flatten')(x)
        flat_shape = x.shape[1:]

        # Now we add the dense layers
        for i, units in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'encoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # Now reshape back to last_shape
        x = keras.layers.Dense(units=np.prod(reshape_shape), activation=activation,
                               name='encoder_dense_final')(x)

        # a dummy layer just to name it latent space
        x = keras.layers.Lambda(lambda x: x, name='LatentSpace')(x)
        self.encoder = keras.Model(inputs=inputs, outputs=x, name='encoder')


        x = keras.layers.Reshape(target_shape=reshape_shape, name='Reshape')(x)
        # Now with transpose convolutions we go back to the original size

        for i, f in enumerate(filters[::-1]):
            x = keras.layers.Conv2DTranspose(
                filters=f, kernel_size=kernel_size[-i-1],
                strides=strides[-i-1],
                use_bias=use_bias, padding=conv_padding, 
                activation=activation,
                name=f'CNN_Transpose_{i+1}')(x)

        # final convolution to get the right number of channels
        x = keras.layers.Conv2DTranspose(filters=1, kernel_size=final_kernel_size,
                                         strides=1, use_bias=use_bias, padding='same', activation=final_activation,
                                         name=f'CNN_Transpose_Final')(x)

        before_padding = x
        # Add zero padding
        padding = calculate_padding(
            input_shape=before_padding.shape[1:], target_shape=input_shape)
        outputs = keras.layers.ZeroPadding2D(
            padding=padding, name='Padding')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=output_name)

        # assert model.layers[-1].output_shape[1:] == input_shape

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model


# model definition
class AutoEncoderTranspose(BaseModel):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, output_name='autoencoder',
                 input_shape=(128, 128, 1), dense_layers=[7],
                 decoder_dense_layers=[],
                 cropping=[[0, 0], [0, 0]],
                 filters=[8, 16, 32],  kernel_size=3, conv_padding='same',
                 strides=[2, 2], 
                 enc_activation='relu',
                 dec_activation='relu',
                 conv_activation='relu',
                 alpha=0.1,
                 final_activation='linear', final_kernel_size=3,
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='valid',
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=True, conv_batchnorm=False,
                 dense_batchnorm=False, **kwargs):
        super().__init__()

        self.output_name = output_name
        self.inputShape = input_shape
        # the kernel_size can be a single int or a list of ints
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(strides[0], int):
            strides = [strides for _ in filters]
        assert len(strides) == len(filters)

        # set the input size
        inputs = keras.Input(shape=input_shape, name='Input')

        # this is the autoencoder case
        # crop the edges
        cropped = keras.layers.Cropping2D(
            cropping=cropping, name='Crop')(inputs)
        x = cropped

        # For evey Convolutional layer
        for i, f in enumerate(filters):
            # Add the Convolution
            x = keras.layers.Conv2D(
                filters=f, kernel_size=kernel_size[i], strides=strides[i],
                use_bias=use_bias, padding=conv_padding, activation=conv_activation,
                name=f'CNN_{i+1}')(x)

            # Apply batchnormalization
            if conv_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Optional pooling after the convolution
            if pooling == 'Max':
                x = keras.layers.MaxPooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'MaxPooling_{i+1}')(x)
            elif pooling == 'Average':
                x = keras.layers.AveragePooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'AveragePooling_{i+1}')(x)

        # we have reached the latent space
        last_shape = x.shape[1:]
        x = keras.layers.Flatten(name='Flatten')(x)
        flat_shape = x.shape[1:]
        # Now we add the dense layers
        for i, units in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=enc_activation,
                                   name=f'encoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'enc_dropout_{i+1}')(x)

        # a dummy layer just to name it latent space
        x = keras.layers.Lambda(lambda x: x, name='LatentSpace')(x)
        self.encoder = keras.Model(inputs=inputs, outputs=x, name='encoder')

        # Now we add the decoder dense layers
        for i, units in enumerate(decoder_dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=dec_activation,
                                   name=f'decoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dec_dropout_{i+1}')(x)

        # Now reshape back to last_shape
        x = keras.layers.Dense(units=np.prod(flat_shape), activation=conv_activation,
                               name='decoder_dense_final')(x)

        x = keras.layers.Reshape(target_shape=last_shape, name='Reshape')(x)
        # Now with transpose convolutions we go back to the original size

        for i, f in enumerate(filters[::-1]):
            x = keras.layers.Conv2DTranspose(
                filters=f, kernel_size=kernel_size[-i-1],
                strides=strides[-i-1],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_Transpose_{i+1}')(x)

        # final convolution to get the right number of channels
        x = keras.layers.Conv2DTranspose(filters=1, kernel_size=final_kernel_size,
                                         strides=1, use_bias=use_bias, padding='same',
                                         name=f'CNN_Transpose_Final')(x)

        x = keras.layers.Activation(activation=final_activation,
                                    name='final_activation')(x)
        before_padding = x
        # Add zero padding
        padding = calculate_padding(
            input_shape=before_padding.shape[1:], target_shape=input_shape)
        outputs = keras.layers.ZeroPadding2D(
            padding=padding, name='Padding')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=output_name)

        # assert model.layers[-1].output_shape[1:] == input_shape

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model


# model definition
class FeatureExtractor(BaseModel):
    def __init__(self, input_shape, output_features=1,
                 output_name='feature_extractor',
                 dense_layers=[256, 64], activation='relu',
                 final_activation='linear',
                 dropout=0.0, learning_rate=1e-3, loss='mse',
                 metrics=[], use_bias=True, batchnorm=False,
                 **kwargs):
        super().__init__()

        self.output_name = output_name

        # set the input size
        inputs = keras.Input(shape=input_shape, name='Input')
        x = inputs

        # Now we add the dense layers
        for i, units in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'dense_{i+1}', use_bias=use_bias)(x)

            # Apply batchnormalization
            if batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # Add the final layer
        outputs = keras.layers.Dense(units=output_features,
                                     activation=final_activation,
                                     name='dense_final', use_bias=use_bias)(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=output_name)

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=[])

        self.model = model


def downsample(filters, kernel_size, apply_batchnorm=True, apply_dropout=0.0,
               use_bias=False, strides=2, padding='same', activation='relu',
               name=None):

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                               use_bias=use_bias))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout > 0 and apply_dropout < 1.0:
        result.add(tf.keras.layers.Dropout(apply_dropout))

    if activation == 'leakyrelu':
        result.add(tf.keras.layers.LeakyReLU())
    elif activation == 'relu':
        result.add(tf.keras.layers.ReLU())

    return result


def upsample(filters, kernel_size, apply_dropout=0.5, apply_batchnorm=True,
             strides=2, padding='same', activation='relu',
             use_bias=False, name=None):

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides,
                                        padding=padding,
                                        use_bias=use_bias))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout > 0 and apply_dropout < 1.0:
        result.add(tf.keras.layers.Dropout(apply_dropout))

    if activation == 'leakyrelu':
        result.add(tf.keras.layers.LeakyReLU())
    elif activation == 'relu':
        result.add(tf.keras.layers.ReLU())

    return result


class Patches(layers.Layer):
    def __init__(self, patch_size, crop_size=0):
        super().__init__()
        self.patch_size = patch_size
        if isinstance(self.patch_size, int):
            self.patch_size = [self.patch_size] * 2
        self.crop_size = crop_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "crop_size": self.crop_size,
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        if self.crop_size > 0:
            images = images[:, self.crop_size:-self.crop_size,
                            self.crop_size:-self.crop_size, :]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # print(patches.shape)
        patch_dims = patches.shape[-1]
        # print('patch_dims', patch_dims)
        patches = tf.reshape(
            patches, [batch_size, patches.shape[1] * patches.shape[2], patch_dims])
        return patches

# Patch encoding layer


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection,
        })
        return config

# MLP with dropout


def mlp(x, hidden_units, dropout_rate, activation='relu'):
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class EncoderSingleViT(keras.Model):
    def __init__(self, output_name, input_shape=(128, 128, 1), cropping=0,
                 patch_size=5, projection_dim=16, transformer_layers=4, num_heads=12,
                 transformer_units=[256, 64], mlp_head_units=[256, 64],
                 dropout_attention=0.1, dropout_mlp=0.1, dropout_representation=0.5, dropout_final=0.5,
                 learning_rate=0.001, loss='mse', activation='relu',
                 final_activation='linear', optimizer='adam',
                 **kwargs):
        super().__init__()

        self.output_name = output_name
        self.inputShape = input_shape
        inputs = layers.Input(shape=input_shape)

        # Create patches.
        patches = Patches(patch_size, cropping)(inputs)

        # Encode patches.
        encoded_patches = PatchEncoder(
            patches.shape[1], projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=dropout_attention)(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units,
                     dropout_rate=dropout_mlp, activation=activation)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(
            epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(dropout_representation)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units,
                       dropout_rate=dropout_final)
        # Classify outputs.
        outputs = layers.Dense(
            1, activation=final_activation, name=output_name)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=outputs)
        self.model = model

        # Also initialize the optimizer and compile the model
        if optimizer == 'adam':
            model.compile(optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate), loss=loss)
        elif optimizer == 'sgd':
            model.compile(optimizer=keras.optimizers.SGD(
                learning_rate=learning_rate), loss=loss)
        elif optimizer == 'adamw':
            model.compile(optimizer=keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate), loss=loss)

        # return model
    def predict(self, waterfall):
        latent = self.model(waterfall)
        return latent

    def load(self, weights_file):
        self.model = keras.models.load_model(weights_file)

    def save(self, weights_file):
        self.model.save(weights_file)


class EncoderSingle(keras.Model):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, output_name, input_shape=(128, 128, 1), dense_layers=[1024, 256, 64],
                 filters=[8, 16, 32], cropping=[[0, 0], [0, 0]], kernel_size=3,
                 strides=[2, 2], activation='relu',
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='same',
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=True, batchnorm=False,
                 conv_batchnorm=False, conv_padding='same',
                   **kwargs):
        super().__init__()

        self.output_name = output_name
        self.inputShape = input_shape
        # the kernel_size can be a single int or a list of ints
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(strides[0], int):
            strides = [strides for _ in filters]
        assert len(strides) == len(filters)

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
                filters=f, kernel_size=kernel_size[i], strides=strides[i],
                use_bias=use_bias, padding=conv_padding, activation=activation,
                name=f'{output_name}_CNN_{i+1}')(x)

            # Apply batchnormalization
            if conv_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Apply the activation function
            # x = keras.activations.get(activation)(x)

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
            
            # Apply batchnormalization
            if batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

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

    def predict(self, waterfall):
        latent = self.model(waterfall)
        return latent

    def load(self, weights_file):
        self.model = keras.models.load_model(weights_file)

    def save(self, weights_file):
        self.model.save(weights_file)


class EncoderMulti:
    var_names = ['phEr', 'enEr', 'bl',
                 'inten', 'Vrf', 'mu',
                 'VrfSPS']

    # By default, the model does not predict the intensity variable
    def __init__(self, enc_list=[]):
        # If a list of models is provided, use it. Otherwise, default initialize models.
        if len(enc_list) > 0:
            self.model = enc_list
        else:
            self.model = [
                EncoderSingle(output_name='phEr', kernel_size=[3, 3],
                              filters=[4, 8]),
                EncoderSingle(output_name='enEr', cropping=[6, 6]),
                EncoderSingle(output_name='bl', cropping=[12, 12],
                              kernel_size=[(13, 3), (7, 3), (3, 3)]),
                EncoderSingle(output_name='inten'),
                EncoderSingle(output_name='Vrf', cropping=[6, 6],
                              kernel_size=[13, 7, 3]),
                EncoderSingle(output_name='mu', kernel_size=[5, 5, 5]),
                EncoderSingle(output_name='VrfSPS', cropping=[6, 6],
                              kernel_size=[5, 5, 5]),
            ]
        self.inputShape = self.model[0].inputShape

    def load(self, weights_dir):

        weights_dir_files = os.listdir(weights_dir)
        for model in self.model:
            # For every model load parameters from weights dir
            var = model.output_name
            fname = f'encoder_{var}.h5'
            if fname in weights_dir_files:
                model.load(os.path.join(weights_dir, fname))
            else:
                raise FileNotFoundError(
                    f'File {fname} not found in {weights_dir}')

    def save(self, model_path):
        # Save all individual models
        for model in self.model:
            var = model.output_name
            file_path = os.path.join(model_path, f'encoder_{var}.h5')
            model.save(file_path)

    def predict(self, waterfall, unnormalize=False, normalization='minmax'):
        # collect the latents
        latent = tf.concat([m.predict(waterfall)
                            for m in self.model], axis=1)
        # optionally normallize them
        if unnormalize:
            latent = unnormalize_params(
                latent[:, 0], latent[:, 1], latent[:, 2],
                latent[:, 3], latent[:, 4], latent[:, 5],
                latent[:, 6], normalization=normalization)
            latent = np.array(latent).T

        return latent

    def summary(self):
        for model in self.model:
            model.model.summary()
            print()


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


class EncoderDecoderModel:

    def __init__(self):
        # Simply initialize two empty models
        self.encoder = EncoderMulti()
        self.decoder = Decoder()
        self.inputShape = self.encoder.inputShape

    def load(self, weights_dir='', enc_weights_dir='', dec_weights_dir=''):
        # Load encoder parameters
        if not enc_weights_dir:
            assert weights_dir
            enc_weights_dir = os.path.join(weights_dir, 'encoder')
        if not os.path.isdir(enc_weights_dir):
            raise FileNotFoundError(f'Directory not found: {enc_weights_dir}')
        self.encoder.load(enc_weights_dir)

        # Load decoder parameters
        if not dec_weights_dir:
            assert weights_dir
            dec_weights_dir = os.path.join(weights_dir, 'decoder')
        if not os.path.isdir(dec_weights_dir):
            raise FileNotFoundError(f'Directory not found: {dec_weights_dir}')
        self.decoder.load(dec_weights_dir)

    def encode(self, WF, unnormalize=False, normalization='minmax'):
        return self.encoder.predict(WF, unnormalize, normalization)

    def decode(self, latent, turn, unnormalize=False):
        return self.decoder.predict(latent, turn, unnormalize)

    def predict(self, WF, turn, unnormalize=False, normalization='minmax'):
        latent = self.encode(WF)
        PS = self.decode(latent, turn)
        if unnormalize:
            latent = unnormalize_params(
                latent[:, 0], latent[:, 1], latent[:, 2],
                latent[:, 3], latent[:, 4], latent[:, 5],
                latent[:, 6], normalization=normalization)
            latent = np.array(latent).T
            PS = unnormalizeIMG(PS)
        return latent, PS

    def save(self, model_path):
        # First save the encoder
        enc_target_dir = os.path.join(model_path, 'encoder')
        os.makedirs(enc_target_dir, exist_ok=True)
        self.encoder.save(enc_target_dir)

        # Then save the decoder
        dec_target_dir = os.path.join(model_path, 'decoder')
        os.makedirs(dec_target_dir, exist_ok=True)
        self.decoder.save(dec_target_dir)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()


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


class EncoderOld(keras.Model):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, input_shape=(128, 128, 1), dense_layers=[1024, 256, 64],
                 filters=[8, 16, 32], cropping=[[0, 0], [0, 0]], kernel_size=3,
                 strides=[2, 2], activation='relu',
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='same',
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=True, batchnorm=False,
                 conv_batchnorm=False, conv_padding='same',
                 **kwargs):
        super().__init__()

        self.inputShape = input_shape
        # the kernel_size can be a single int or a list of ints
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(strides[0], int):
            strides = [strides for _ in filters]
        assert len(strides) == len(filters)

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
                filters=f, kernel_size=kernel_size[i], strides=strides[i],
                use_bias=use_bias, padding=conv_padding, activation=activation,
                name=f'CNN_{i+1}')(x)

            # Apply batchnormalization
            if conv_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Apply the activation function
            # x = keras.activations.get(activation)(x)

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
        x = keras.layers.Flatten(name=f'Flatten')(x)
        # For each optional dense layer
        for i, layer in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(layer, activation=activation,
                                   name=f'Dense_{i+1}')(x)

            # Apply batchnormalization
            if batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(
                    dropout, name=f'Dropout_{i+1}')(x)

        # Add the final layers, one for each output
        outputs = (x)

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


class TomoscopeOld(keras.Model):
    """The model should take as input the waterfall and output Phasespace.
    It should look like an autoencoder, but the output dimension should be different (larger)
    """

    def __init__(self, output_name='tomoscope', input_shape=(128, 128, 1),
                 output_turns=1, cropping=[[0, 0], [0, 0]],
                 enc_dense_layers=[1024, 256, 64], enc_filters=[8, 16, 32],
                 dec_dense_layers=[256, 1024], dec_filters=[32, 16, 1],
                 enc_kernel_size=3, dec_kernel_size=3,
                 enc_strides=[2, 2], dec_strides=[2, 2],
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
        cropped = keras.layers.Cropping2D(
            cropping=cropping, name='crop')(inputs)
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
        x = keras.layers.Dense(units=np.prod(
            t_shape), activation=dec_activation, name='decoder_expand')(x)
        x = keras.layers.Reshape(target_shape=t_shape,
                                 name='decoder_reshape')(x)

        # For every Convolutional layer
        for i, f in enumerate(dec_filters[:]):
            # Add the Convolution
            x = keras.layers.Conv2DTranspose(
                filters=dec_filters[i], kernel_size=dec_kernel_size[i], strides=dec_strides[i],
                activation=dec_activation, name=f'decoder_cnn_{i+1}', padding='same')(x)

        outputs = keras.layers.Activation(
            activation=final_activation, name='final_activation')(x)

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
        assert os.path.exists(
            weights_file), f'Weights file {weights_file} does not exist'
        self.model = keras.models.load_model(weights_file, compile=compile)

    def save(self, weights_file):
        self.model.save(weights_file)


class Tomoscope(keras.Model):
    """The model should take as input the waterfall and output Phasespace.
    It should look like an autoencoder, but the output dimension should be different (larger)
    """

    def __init__(self, output_name='tomoscope', input_shape=(128, 128, 1),
                 output_turns=1, cropping=[[0, 0], [0, 0]],
                 enc_filters=[8, 16, 32],
                 dec_filters=[32, 16, 1],
                 enc_kernel_size=3, dec_kernel_size=3,
                 enc_strides=[2, 2], dec_strides=[2, 2],
                 enc_activation='relu', dec_activation='relu',
                 final_activation='tanh',
                 enc_batchnorm=False, dec_batchnorm=True,
                 enc_dropout=0.0, dec_dropout=0.0,
                 enc_conv_padding='same', dec_conv_padding='same',
                 learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=False,
                 **kwargs):
        super().__init__()

        self.output_name = output_name

        # The output shape will be (128, 128, output_turns)
        output_shape = (input_shape[0], input_shape[1], output_turns)

        assert len(enc_filters) == len(dec_filters) + 1

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

        # Construct the encoder

        # set the input size
        inputs = keras.Input(shape=input_shape, name='input')
        # crop the edges
        cropped = keras.layers.Cropping2D(
            cropping=cropping, name='crop')(inputs)
        x = cropped

        down_stack = []
        for i, filters in enumerate(enc_filters):
            down_stack.append(downsample(filters, enc_kernel_size[i], strides=enc_strides[i],
                                         use_bias=use_bias, padding=enc_conv_padding, activation=enc_activation,
                                         apply_batchnorm=enc_batchnorm, apply_dropout=enc_dropout,
                                         name=f'encoder_cnn_{i+1}'))


        up_stack = []

        for i, filters in enumerate(dec_filters):
            up_stack.append(upsample(filters, kernel_size=dec_kernel_size[i], strides=dec_strides[i],
                                     activation=dec_activation, use_bias=use_bias, apply_dropout=dec_dropout,
                                     apply_batchnorm=dec_batchnorm, padding=dec_conv_padding, name=f'decoder_cnn_{i+1}'))

        last = tf.keras.layers.Conv2DTranspose(output_turns, kernel_size=4,
                                               strides=2,
                                               padding='same',
                                               name=f'decoder_cnn_{len(dec_filters)+1}',
                                               activation=final_activation)  # (batch_size, 128, 128, 1)


        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for idx, (up, skip) in enumerate(zip(up_stack, skips)):
            x = up(x)
            assert x.shape[:-1] == skip.shape[:-1], f'idx: {idx}, {x.shape}, {skip.shape}'
            x = tf.keras.layers.Concatenate(name=f'concat_{idx+1}')([x, skip])

        x = last(x)
        
        assert x.shape[1:] == output_shape, f'{x.shape[1:]}, {output_shape}'


        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model = keras.Model(inputs=inputs, outputs=x)

        if loss == 'custom':
            loss = custom_loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # # For evey Convolutional layer
        # for i, f in enumerate(enc_filters):
        #     # Add the Convolution
        #     x = keras.layers.Conv2D(
        #         filters=f, kernel_size=enc_kernel_size[i], strides=enc_strides[i],
        #         use_bias=use_bias, padding=enc_conv_padding,
        #         name=f'encoder_cnn_{i+1}')(x)

        #     # Apply batchnormalization
        #     if batchnorm:
        #         x = tf.keras.layers.BatchNormalization()(x)

        #     # Apply the activation function
        #     x = keras.activations.get(enc_activation)(x)

        #     # Optional pooling after the convolution
        #     if enc_pooling == 'Max':
        #         x = keras.layers.MaxPooling2D(
        #             pool_size=enc_pooling_size, strides=enc_pooling_strides,
        #             padding=enc_pooling_padding, name=f'encdoer_maxpooling_{i+1}')(x)
        #     elif enc_pooling == 'Average':
        #         x = keras.layers.AveragePooling2D(
        #             pool_size=enc_pooling_size, strides=enc_pooling_strides,
        #             padding=enc_pooling_padding, name=f'encdoer_averagepooling_{i+1}')(x)

        # Flatten after the convolutions
        # x = keras.layers.Flatten(name=f'encoder_flatten')(x)
        # # For each optional dense layer
        # for i, layer in enumerate(enc_dense_layers):
        #     # Add the layer
        #     x = keras.layers.Dense(layer, activation=enc_activation,
        #                            name=f'encoder_dense_{i+1}')(x)
        #     # Add dropout optionally
        #     if enc_dropout > 0 and enc_dropout < 1:
        #         x = keras.layers.Dropout(
        #             enc_dropout, name=f'encoder_dropout_{i+1}')(x)

        # # Middle layer
        # # middle = x
        # # print('middle shape:', middle.shape)

        # # The dimension should be equal to (middle_dense_layer, )

        # # From this, we want to go to (128, 128, output_turns)

        # # Generate the inverse model (encoder) to find the t_shape
        # temp = keras.Sequential()
        # temp.add(keras.Input(shape=output_shape, name='temp_input'))

        # # temp.add(keras.layers.Conv2D(filters=dec_filters[-1], padding='same',
        # #                              strides=dec_strides[-1], kernel_size=dec_kernel_size[-1]))

        # for i in np.arange(len(dec_filters)-1, -1, -1):
        #     temp.add(keras.layers.Conv2D(filters=dec_filters[i], padding='same',
        #                                  strides=dec_strides[i], kernel_size=dec_kernel_size[i]))
        # # print('Output shape:', output_shape)
        # # print(temp.summary())
        # t_shape = temp.layers[-1].output_shape[1:]

        # del temp

        # # print('t_shape:', t_shape)

        # # Now generate the decoder
        # # For each optional dense layer
        # for i, layer in enumerate(dec_dense_layers):
        #     # Add the layer
        #     x = keras.layers.Dense(layer, activation=dec_activation,
        #                            name=f'decoder_dense_{i+1}')(x)
        #     # Add dropout optionally
        #     if dec_dropout > 0 and dec_dropout < 1:
        #         x = keras.layers.Dropout(
        #             dec_dropout, name=f'decoder_dropout_{i+1}')(x)

        # # extend to needed t_shape and reshape
        # x = keras.layers.Dense(units=np.prod(
        #     t_shape), activation=dec_activation, name='decoder_expand')(x)
        # x = keras.layers.Reshape(target_shape=t_shape,
        #                          name='decoder_reshape')(x)

        # # For every Convolutional layer
        # for i, f in enumerate(dec_filters[:]):
        #     # Add the Convolution
        #     x = keras.layers.Conv2DTranspose(
        #         filters=dec_filters[i], kernel_size=dec_kernel_size[i], strides=dec_strides[i],
        #         activation=dec_activation, name=f'decoder_cnn_{i+1}', padding='same')(x)

        # outputs = keras.layers.Activation(
        #     activation=final_activation, name='final_activation')(x)

        # print('outputs shape:', outputs.shape)

        # Also initialize the optimizer and compile the model
        # optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # self.model = keras.Model(inputs=inputs, outputs=outputs)

        # if loss == 'custom':
        #     loss = custom_loss
        # self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def predict(self, waterfall):
        tomography = self.model(waterfall)
        return tomography

    def load(self, weights_dir, compile=False):
        weights_file = os.path.join(weights_dir, 'tomoscope.h5')
        assert os.path.exists(
            weights_file), f'Weights file {weights_file} does not exist'
        self.model = keras.models.load_model(weights_file, compile=compile)

    def save(self, weights_file):
        self.model.save(weights_file)
