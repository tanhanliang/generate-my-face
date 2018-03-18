"""
Runs everything.
"""

import utility.make_training_data as make
import models.models as models
import numpy as np
import models.parameters as params
from keras.layers import Input
from keras.models import Model
import keras.optimizers as opt
from PIL import Image


def build_gan():
    """
    Builds the models used in the GAN.

    :return: a tuple of 3 keras Models
    """

    # Build the models
    discriminator = models.build_discriminator()
    optimiser = opt.adam(lr=0.002)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimiser,
                          metrics=['accuracy'])
    generator = models.build_generator()
    discriminator.trainable = False

    # Connect the generator to the discriminator
    gan_input = Input(shape=params.NOISE_SHAPE)
    generated_img = generator(gan_input)
    discrim_out = discriminator(generated_img)

    # Build and compile the full GAN
    combined_model = Model(gan_input, discrim_out)
    optimiser = opt.adam(lr=0.002)
    combined_model.compile(loss='binary_crossentropy',
                           optimizer=optimiser,
                           metrics=['accuracy'])
    return discriminator, generator, combined_model


def train(discriminator, generator, combined_model, epochs, save_interval):
    """
    Runs the GAN.

    :param discriminator: A keras Model
    :param generator:A keras Model
    :param combined_model:A keras Model
    :param epochs: Number of 'back and forth' iterations of training the discriminator,
    then training the generator.
    :param save_interval: The interval at which we save images.
    :return: Nothing.
    """

    x, y = make.get_training_data()

    half_batch = int(y.shape[0]/2)
    batch_size = y.shape[0]

    for epoch in range(epochs):
        # To reduce the change of overfitting, get random images
        indices = np.random.randint(0, y.shape[0], half_batch)
        real_images = x[indices]

        # Train the discriminator first
        noise = np.random.normal(0, 1, (half_batch,) + params.NOISE_SHAPE)
        generated_images = generator.predict(noise)
        test_x = np.concatenate((real_images, generated_images))
        test_y = np.concatenate((np.ones(half_batch), np.zeros(half_batch)))
        test_x, test_y = make.shuffle_datasets(test_x, test_y)

        # discrim_loss_r = discriminator.train_on_batch(real_images, np.ones(half_batch))
        # discrim_loss_g = discriminator.train_on_batch(generated_images, np.zeros((half_batch, 1)))

        # Now train the generator
        noise = np.random.normal(0, 1, (batch_size,) + params.NOISE_SHAPE)
        # comb_loss = combined_model.train_on_batch(noise, np.ones((batch_size, 1)))

        # print("Epoch %d [D_Loss_Real: %f Acc_Real: %f ] [D_Loss_Fake: %f Acc_Fake: %f] [Comb_Loss: %f Acc_Comb: %f]"
        #       % (epoch, discrim_loss_r[0], discrim_loss_r[1], discrim_loss_g[0], discrim_loss_g[1], comb_loss[0], comb_loss[1]))
        print("Epoch " + str(epoch) + " Training Discriminator")
        discriminator.fit(test_x, test_y, 50, 1)
        print("Epoch " + str(epoch) + " Training Generator")
        combined_model.fit(noise, np.ones(batch_size), 50, 1)

        if epoch % save_interval == 0:
            save_image(generator, epoch)


def save_image(generator, epoch):
    """
    Feeds some noise input to the generator, then saves the resulting output as an image.

    :param generator: A keras Model
    :param epoch: An integer
    :return: Nothing
    """

    # The generator model expects a 4-d input, with the first dimension being training examples
    noise = np.random.normal(0, 1, (1,) + params.NOISE_SHAPE)
    gen_img = generator.predict(noise)
    gen_img = gen_img*127.5 + 127.5
    gen_img = gen_img.astype(np.uint8)

    # Now we have to reshape the output back to 3 dimensions
    gen_img = gen_img.reshape(params.IMG_SHAPE)
    img = Image.fromarray(gen_img, "RGB")
    img_name = "outputs/save_" + str(epoch) + ".png"
    img.save(img_name)
