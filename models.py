import tensorflow as tf
import numpy as np
from utils import normalize_params

class extendedCED(tf.keras.Model):

    def __init__(self, latent_dim, additional_latent_dim, input_shape, filters):
        super(extendedCED, self).__init__()
        self.latent_dim = latent_dim
        self.inputShape = input_shape
        self.additional_latent_dim = additional_latent_dim

        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(
            input_shape=self.inputShape))
        for f in filters:
            self.encoder.add(tf.keras.layers.Conv2D(
                filters=f, kernel_size=3, strides=(2, 2), activation='relu'))
        t_shape = self.encoder.layers[-1].output_shape[1:]
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(latent_dim))

        self.extender = tf.keras.Sequential()
        self.extender.add(tf.keras.layers.InputLayer(
            input_shape=self.latent_dim + self.additional_latent_dim))

        t_shape = (t_shape[0]+1, t_shape[1]+1, int(t_shape[2]/2))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer(
            input_shape=(self.latent_dim + self.additional_latent_dim)))
        self.decoder.add(tf.keras.layers.Dense(
            units=np.prod(t_shape), activation=tf.nn.relu))
        self.decoder.add(tf.keras.layers.Reshape(target_shape=t_shape))
        for f in reversed(filters):
            self.decoder.add(tf.keras.layers.Conv2DTranspose(
                filters=f, kernel_size=3, strides=2, padding='same', activation='relu'))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same'))

        self.encoder.summary()
        self.extender.summary()
        self.decoder.summary()

    @tf.function
    def encode(self, x):
        return self.encoder(x)

    @tf.function
    def decode(self, z):
        return self.decoder(z)

    @tf.function
    def extend(self, encoded_latent_vec, turn_normalized):
        return self.extender(tf.keras.layers.Concatenate()([encoded_latent_vec,
                                                            tf.transpose(tf.keras.layers.Flatten()(turn_normalized))]))

    @tf.function
    def predictPS(self, T_images_input, turn_normalized):
        z = self.encode(T_images_input)
        return self.decode(self.extend(z, turn_normalized)), z


@tf.function
def compute_mse_physical_loss(model, turn_normalized, T_image, PS_image, phErs, enErs, bls, intens, Vrf, mu):
    phErs_norm, enErs_norm, bls_norm, intens_norm, Vrf_norm, mu_norm = normalize_params(
        phErs, enErs, bls, intens, Vrf, mu)
    predicted_beam_logit, latents = model.predictPS(
        T_image, turn_normalized, training=True)
    return tf.keras.metrics.mse(tf.keras.backend.flatten(PS_image),
                                tf.keras.backend.flatten(predicted_beam_logit)),\
        tf.keras.metrics.mse(tf.keras.backend.flatten(latents),
                             tf.keras.backend.flatten(tf.transpose(tf.convert_to_tensor([phErs_norm,
                                                                                         enErs_norm,
                                                                                         bls_norm,
                                                                                         intens_norm,
                                                                                         Vrf_norm,
                                                                                         mu_norm]))))
