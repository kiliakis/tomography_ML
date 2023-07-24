# Train the ML model

from utils import sample_files, tomoscope_files_to_tensors
from utils import fast_tensor_load_encdec

import time
import shutil
import tensorflow as tf
import yaml
import os
import numpy as np
from datetime import datetime
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

parser = argparse.ArgumentParser(description='Train the tomoscope models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')

# Initialize parameters
data_dir = './tomo_data/datasets_tomoscope_TF_24-03-23'

timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

# Data specific
DATA_LOAD_METHOD = 'FAST_TENSOR' # Can be TENSOR, FAST_TENSOR or DATASET

# Train specific
train_cfg = {
    'epochs': 10000, 'output_turns': 1,
    # 'gen_down_layers': [(32, 4, False), (64, 4, True), (128, 4, True),
    #                 (256, 4, True), (256, 4, True), (256, 4, True),
    #                 (256, 4, True)],
    # 'gen_up_layers': [(256, 4, 0.5), (256, 4, 0.5), (256, 4, 0.5),
    #                (128, 4, 0), (64, 4, 0), (32, 4, 0)],

    'gen_down_layers': [(8, 4, False), (16, 4, True), (32, 4, True),
                    (64, 4, True)],
    'gen_up_layers': [(32, 4, 0), (16, 4, 0), (8, 4, 0)],

    'gen_final_activation': 'linear',
    'disc_down_layers': [(8, 4, False), (16, 4, True), (32, 4, True)],
    'down_layers_bias': False, 'up_layers_bias': False,
    'gen_learning_rate': 2e-4,
    'disc_learning_rate': 2e-4,

    'dataset%': 1,
    'normalization': 'minmax', 'img_normalize': 'off',
    'ps_normalize': 'off',
    'batch_size': 1,
    'output_channels': 1,
    'LAMBDA': 100,
}



def downsample(filters, size, apply_batchnorm=True, use_bias=False, 
               name=None):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=use_bias))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=0.5, use_bias=False, name=None):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=use_bias))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout > 0 and apply_dropout < 1.0:
        result.add(tf.keras.layers.Dropout(apply_dropout))

    result.add(tf.keras.layers.ReLU())

    return result

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    gan_loss = loss_object(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Generator(input_shape, output_channels, down_layers, up_layers, final_activation='tanh'):
    inputs = tf.keras.layers.Input(shape=input_shape)

    assert len(down_layers) == len(up_layers) + 1

    down_stack = []
    for i, (filters, kernel_size, batchnorm) in enumerate(down_layers):
        down_stack.append(downsample(filters, kernel_size, apply_batchnorm=batchnorm, name=f'gen_down_{i+1}'))

    # down_stack = [
    #     downsample(32, 4, apply_batchnorm=False, name='encoder_1'),  # (batch_size, 64, 64, 32)
    #     downsample(64, 4, name='encoder_2'),  # (batch_size, 32, 32, 64)
    #     downsample(128, 4, name='encoder_3'),  # (batch_size, 16, 16, 128)
    #     downsample(256, 4, name='encoder_4'),  # (batch_size, 8, 8, 256)
    #     downsample(256, 4, name='encoder_5'),  # (batch_size, 4, 4, 256)
    #     downsample(256, 4, name='encoder_6'),  # (batch_size, 2, 2, 256)
    #     downsample(256, 4, name='encoder_7'),  # (batch_size, 1, 1, 256)
    # ]

    up_stack = []

    for i, (filters, kernel_size, dropout) in enumerate(up_layers):
        up_stack.append(upsample(filters, kernel_size, apply_dropout=dropout, name=f'gen_up_{i+1}'))

    # up_stack = [
    #     upsample(256, 4, apply_dropout=dropout, name='decoder_1'),  # (batch_size, 2, 2, 256)
    #     upsample(256, 4, apply_dropout=dropout, name='decoder_2'),  # (batch_size, 4, 4, 256)
    #     upsample(256, 4, apply_dropout=dropout, name='decoder_3'),  # (batch_size, 8, 8, 256)
    #     upsample(128, 4, name='decoder_4'),  # (batch_size, 16, 16, 128)
    #     upsample(64, 4, name='decoder_5'),  # (batch_size, 32, 32, 64)
    #     upsample(32, 4, name='decoder_6'),  # (batch_size, 64, 64, 32)
    # ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           name=f'gen_up_{len(up_layers)+1}',
                                           activation=final_activation)  # (batch_size, 128, 128, 1)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for idx, (up, skip) in enumerate(zip(up_stack, skips)):
        x = up(x)
        x = tf.keras.layers.Concatenate(name=f'gen_concat_{idx+1}')([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator(input_shape, output_shape, down_layers, use_bias=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=output_shape, name='target_image')

    # (batch_size, 128, 128, channels*2)
    x = tf.keras.layers.concatenate([inp, tar])

    for i, (filters, kernel_size, batchnorm) in enumerate(down_layers):
        x = downsample(filters, kernel_size, apply_batchnorm=batchnorm, name=f'disc_down_{i+1}')(x)

    # down1 = downsample(32, 4, False, name='discriminator_1')(x)  # (batch_size, 64, 64, 32)
    # down2 = downsample(64, 4, name='discriminator_2')(down1)  # (batch_size, 32, 32, 64)
    # down3 = downsample(128, 4, name='discriminator_3')(down2)  # (batch_size, 16, 16, 128)

    zero_pad1 = tf.keras.layers.ZeroPadding2D(name='disc_zeropad_1')(x)  # (batch_size, 18, 18, 128)
    conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                  kernel_initializer=initializer,
                                  name=f'disc_down_{len(down_layers)+1}',
                                  use_bias=use_bias)(zero_pad1)  # (batch_size, 15, 15, 256)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D(name='disc_zeropad_2')(
        leaky_relu)  # (batch_size, 17, 17, 256)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  name='disc_down_last',
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 14, 14, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generate_images(model, test_input, tar, show=False, save_path=None):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


@tf.function
def train_step(generator, gen_optimizer, discriminator, disc_optimizer, input_image, target, 
               step, summary_writer, plot_every=1000, LAMBDA=100):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator(
            [input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target, LAMBDA=LAMBDA)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//plot_every)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//plot_every)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//plot_every)
        tf.summary.scalar('disc_loss', disc_loss, step=step//plot_every)



def fit(generator, gen_optimizer, discriminator, disc_optimizer, train_ds, valid_ds, steps, 
        checkpoint, checkpoint_prefix, summary_writer, plots_dir,
        plot_every=1000, checkpoint_every=5000, LAMBDA=100):

    example_input, example_target = next(iter(valid_ds.take(1)))

    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % plot_every == 0:

            if step != 0:
                print(
                    f'Time taken for {plot_every} steps: {time.time()-start:.2f} sec\n')

            start = time.time()
            generate_images(generator, example_input, example_target, 
                            show=False, save_path=os.path.join(plots_dir, f'visual_eval_{step}.png'))

            print(f"Step: {step//plot_every}k")

        train_step(generator, gen_optimizer, discriminator, disc_optimizer, input_image, 
                   target, step, summary_writer, plot_every, LAMBDA=LAMBDA)

        # Training step
        if (step+1) % (max(plot_every//100, 1)) == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every some steps
        if (step + 1) % checkpoint_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)



if __name__ == '__main__':

    args = parser.parse_args()
    # If input config file is provided, read input from config file
    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['pix2pix']
        timestamp = input_config['timestamp']

    print('\n---- Configuration: ----\n')
    for k, v in train_cfg.items():
        print(k, v)

    # Initialize directories
    trial_dir = os.path.join('./trials/', timestamp)
    weights_dir = os.path.join(trial_dir, 'weights')
    plots_dir = os.path.join(trial_dir, 'plots')
    log_dir = os.path.join(trial_dir, 'logs')
    cache_dir = os.path.join(trial_dir, 'cache')

    print('\n---- Using directory: ', trial_dir, ' ----\n')

    # Initialize GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    device_to_use = 0

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            # tf.config.experimental.set_memory_growth(gpus[device_to_use], True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[device_to_use],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12*1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print('No GPU available, using the CPU')

    # Initialize train/ test / validation paths
    ML_dir = os.path.join(data_dir, 'ML_data')
    assert os.path.exists(ML_dir)

    TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')

    # create the directory to store the results
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # os.makedirs(cache_dir, exist_ok=True)

    try:
        start_t = time.time()
        # Create the datasets
        if DATA_LOAD_METHOD=='TENSOR':
            # First the training data
            file_names = sample_files(
                TRAINING_PATH, train_cfg['dataset%'], keep_every=1)
            print('Number of Training files: ', len(file_names))
            x_train, y_train = tomoscope_files_to_tensors(
                file_names, normalization=train_cfg['normalization'],
                img_normalize=train_cfg['img_normalize'],
                ps_normalize=train_cfg['ps_normalize'], num_turns=train_cfg['output_turns'])

            # Repeat for validation data
            file_names = sample_files(
                VALIDATION_PATH, train_cfg['dataset%'], keep_every=1)
            print('Number of Validation files: ', len(file_names))

            x_valid, y_valid = tomoscope_files_to_tensors(
                file_names, normalization=train_cfg['normalization'],
                img_normalize=train_cfg['img_normalize'],
                ps_normalize=train_cfg['ps_normalize'], num_turns=train_cfg['output_turns'])
            
            print('x_train shape: ', x_train.shape)
            print('x_valid shape: ', x_valid.shape)
        
        elif DATA_LOAD_METHOD == 'FAST_TENSOR':
            assert train_cfg['normalization'] == 'minmax'
            assert train_cfg['ps_normalize'] == 'off'
            assert train_cfg['img_normalize'] == 'off'

            TRAINING_PATH = os.path.join(ML_dir, 'tomoscope-training-??.npz')
            VALIDATION_PATH = os.path.join(ML_dir, 'tomoscope-validation-??.npz')

            x_train, turn_train, latent_train, y_train = fast_tensor_load_encdec(
                TRAINING_PATH, train_cfg['dataset%'])
            print('Number of Training files: ', len(y_train))

            x_valid, turn_valid, latent_valid, y_valid = fast_tensor_load_encdec(
                VALIDATION_PATH, train_cfg['dataset%'])
            print('Number of Validation files: ', len(y_valid))

            y_train = y_train[:, :, :, :train_cfg['output_turns']]
            y_valid = y_valid[:, :, :, :train_cfg['output_turns']]

        elif DATA_LOAD_METHOD=='DATASET':
            exit('DATASET method not supported')

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(train_cfg['batch_size'])

        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        valid_ds = valid_ds.batch(1)

        end_t = time.time()
        print(
            f'\n---- Input files have been read, elapsed: {end_t - start_t} ----\n')


        start_t = time.time()
        input_shape = x_train.shape[1:]
        output_shape = y_train.shape[1:]

        # Model instantiation
        generator = Generator(input_shape=input_shape, output_channels=train_cfg['output_channels'],
                              down_layers=train_cfg['gen_down_layers'], up_layers=train_cfg['gen_up_layers'],
                              final_activation=train_cfg['gen_final_activation'])
        print('Generator summary')
        print(generator.summary())

        discriminator = Discriminator(input_shape=input_shape, output_shape=output_shape,
                                      down_layers=train_cfg['disc_down_layers'])
        print('Discriminator summary')
        print(discriminator.summary())


        end_t = time.time()
        print(
            f'\n---- Model has been initialized, elapsed: {end_t - start_t} ----\n')

        print('\n---- Training the model ----\n')

        summary_writer = tf.summary.create_file_writer(log_dir)
        
        generator_optimizer = tf.keras.optimizers.Adam(train_cfg['gen_learning_rate'], beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(train_cfg['disc_learning_rate'], beta_1=0.5)

        checkpoint_prefix = os.path.join(weights_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=generator,
                                        discriminator=discriminator)
        

        start_time = time.time()
        
        fit(generator, generator_optimizer, discriminator, discriminator_optimizer, 
            train_ds, valid_ds, steps=train_cfg['epochs'],
            checkpoint=checkpoint,
            checkpoint_prefix=checkpoint_prefix,
            summary_writer=summary_writer,
            plots_dir=plots_dir,
            LAMBDA=train_cfg['LAMBDA'])

        total_time = time.time() - start_time
        print(
            f'\n---- Training complete, epochs: {train_cfg["epochs"]} total time {total_time} ----\n')

        # Plot training and validation loss
        print('\n---- Plotting loss ----\n')

        # Restoring the latest checkpoint in checkpoint_dir
        checkpoint.restore(tf.train.latest_checkpoint(weights_dir))
        
        # Run the trained model on a few examples from the test set
        idx = 0
        for inp, tar in valid_ds.take(5):
            generate_images(generator, inp, tar, show=False, save_path=os.path.join(plots_dir, f'visual_eval_idx{idx}_{train_cfg["epochs"]}.png'))
            idx+=1

        # save file with experiment configuration
        print('\n---- Saving a summary ----\n')

        config_dict = {}
        config_dict['pix2pix'] = train_cfg.copy()

        config_dict['pix2pix'].update({
            'epochs': train_cfg['epochs'],
            # 'min_train_loss': float(np.min(train_loss_l)),
            # 'min_valid_loss': float(np.min(valid_loss_l)),
            'total_train_time': total_time,
            'used_gpus': len(gpus)
        })

        # save config_dict
        with open(os.path.join(trial_dir, 'pix2pix-summary.yml'), 'w') as configfile:
            yaml.dump(config_dict, configfile, default_flow_style=False)
    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
