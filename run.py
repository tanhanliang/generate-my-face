"""
Runs everything.
"""

import utility.make_training_data as make
import models.gan as models
import numpy as np
import models.parameters as params
from PIL import Image
from utility.troubleshooting import CustomMetrics


def train(gan, epochs, save_interval, callback=None):
    """
    Runs the GAN.

    :param gan: A GAN object
    :param epochs: Number of 'back and forth' iterations of training the discriminator,
    then training the generator.
    :param save_interval: The interval at which we save images.
    :param callback: A keras callback for troubleshooting
    :return: Nothing.
    """

    x = make.get_training_data("images/", ".jpg")

    # batch_size = int(y.shape[0]/2)
    batch_size = 5

    for epoch in range(epochs):
        # To reduce the change of overfitting, get random images
        indices = np.random.randint(0, x.shape[0], batch_size)
        real_images = x[indices]

        # Train the discriminator and generator. Due to the nature of the loss calculations, where
        # the stored value of generator loss is used, even though these two fit statements should
        # be the same as if they were run simultaneously

        noise = np.random.uniform(-1, 1, (batch_size,) + params.NOISE_SHAPE)
        fake_images = gan.generator.predict(noise)
        gan.combined_model.fit(noise, fake_images, batch_size, 1, verbose=0, callbacks=[callback])
        gan.discriminator.fit(real_images, real_images, batch_size, 1, verbose=0, callbacks=[callback])

        gan.print_convergence_measures(epoch, callback)

        if epoch % save_interval == 0:
            save_image(gan.generator, epoch)


def save_image(generator, epoch):
    """
    Feeds some noise input to the generator, then saves the resulting output as an image.

    :param generator: A keras Model
    :param epoch: An integer
    :return: Nothing
    """

    # The generator model expects a 4-d input, with the first dimension being training examples
    noise = np.random.uniform(-1, 1, (1,) + params.NOISE_SHAPE)
    gen_img = generator.predict(noise)
    gen_img = gen_img*127.5 + 127.5
    gen_img = gen_img.astype(np.uint8)

    # Now we have to reshape the output back to 3 dimensions
    gen_img = gen_img.reshape(params.IMG_SHAPE)
    img = Image.fromarray(gen_img, "RGB")
    img_name = "outputs/save_" + str(epoch) + ".png"
    img.save(img_name)


def main():
    gan = models.GAN(0.001, 0, 0.5, 1)
    metrics = ["get_kt", "get_gen_loss", "get_r_reconstr_loss", "get_discrim_loss"]
    train(gan, 20000, 50, CustomMetrics(metrics))


if __name__ == "__main__":
    main()
