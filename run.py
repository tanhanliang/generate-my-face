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


def train(epochs, save_interval):
    """
    Runs the GAN.

    :param epochs: Number of 'back and forth' iterations of training the discriminator,
    then training the generator.
    :param save_interval: The interval at which we save images.
    :return: Nothing.
    """

    x, y = make.get_training_data()

    discriminator = models.build_discriminator()
    discriminator.trainable = False
    generator = models.build_generator()

    gan_input = Input(shape=params.NOISE_SHAPE)
    generated_img = generator(gan_input)
    discrim_out = discriminator(generated_img)

    combined_model = Model(gan_input, discrim_out)
    optimiser = opt.adam(lr=0.002)
    combined_model.compile(loss='binary_crossentropy',
                           optimizer=optimiser,
                           metrics=['accuracy'])

    half_batch = int(y.shape[0]/2)
    batch_size = y.shape[0]

    for epoch in range(epochs):
        # To reduce the change of overfitting, get random images
        indices = np.random.randint(0, y.shape[0], half_batch)
        real_images = x[indices]

        # Train the discriminator first
        noise = np.random.normal(0, 1, (half_batch,) + params.NOISE_SHAPE)
        generated_images = generator.predict(noise)
        discrim_loss_r = discriminator.train_on_batch(real_images, np.ones(half_batch, 1))
        discrim_loss_g = discriminator.train_on_batch(generated_images, np.zeros((half_batch, 1)))

        # Now train the generator
        noise = np.random.normal(0, 1, (batch_size,) + params.NOISE_SHAPE)
        comb_loss = combined_model.train_on_batch(noise, np.ones(batch_size, 1))

        print("Epoch %d [D_Loss_Real: %f Acc_Real: %f ] [D_Loss_Fake: %f Acc_Fake: %f] [Combined_Loss: %f]"
              % (epoch, discrim_loss_r[0], discrim_loss_r[1], discrim_loss_g[0], discrim_loss_g[1], comb_loss))

        if epoch % save_interval == 0:
            save_image(generator, epoch)


def save_image(generator, epoch):
    """
    Feeds some noise input to the generator, then saves the resulting output as an image.

    :param generator: A keras Model
    :param epoch: An integer
    :return: Nothing
    """

    noise = np.random.normal(0, 1, params.NOISE_SHAPE)
    gen_img = generator.predict(noise)
    img = Image.fromarray(gen_img, "RGB")
    img_name = "outputs/save_" + str(epoch) + ".png"
    img.save(img_name)
