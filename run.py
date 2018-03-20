"""
Runs everything.
"""

import utility.make_training_data as make
import models.gan as models
import numpy as np
import models.parameters as params
from keras.layers import Input
from keras.models import Model
import keras.optimizers as opt
from PIL import Image


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

    x, y = make.get_training_data("images/", ".jpg")

    train_diff_factor = 10
    d_batch_size = int(y.shape[0]/train_diff_factor)
    batch_size = d_batch_size*2*train_diff_factor

    for epoch in range(epochs):
        # To reduce the change of overfitting, get random images
        indices = np.random.randint(0, y.shape[0], d_batch_size)
        real_images = x[indices]

        # Get the training data
        noise = np.random.normal(0, 1, (d_batch_size,) + params.NOISE_SHAPE)
        generated_images = generator.predict(noise)
        test_x = np.concatenate((real_images, generated_images))
        test_y = np.concatenate((np.ones(d_batch_size), np.zeros(d_batch_size)))
        test_x, test_y = make.shuffle_datasets(test_x, test_y)

        # Train the discriminator, then the generator
        discriminator.fit(test_x, test_y, batch_size, 1, verbose=0)
        d_metrics = discriminator.evaluate(test_x, test_y, verbose=0)

        noise = np.random.normal(0, 1, (batch_size,) + params.NOISE_SHAPE)
        targets = np.ones(batch_size)
        combined_model.fit(noise, targets, batch_size, 1, verbose=0)
        c_metrics = combined_model.evaluate(noise, targets, verbose=0)

        print("Epoch: %d Discriminator [Loss: %f Acc: %f] Generator [Loss: %f Acc %f]" %
              (epoch, d_metrics[0], d_metrics[1], c_metrics[0], c_metrics[1]))

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


def main():
    d, g, c = build_gan()
    train(d, g, c, 20000, 50)


if __name__ == "__main__":
    main()
