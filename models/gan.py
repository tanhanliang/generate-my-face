"""
Builds the discriminator and the generator.
"""
import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
import models.parameters as params
import numpy as np
from keras.layers import Input
from keras.models import Model


class GAN:
    def __init__(self, lmbda, k_t, gamma, norm):
        """
        Builds and wires up the models.
        See https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/

        :param lmbda: The learning rate for k_t
        :param k_t: The adaptive term that balances the losses of the discriminator and
        generator automatically
        :param gamma: The tradeoff between image diversity and quality
        :param norm: L1 norm or L2 norm. An integer in {1, 2}
        """
        self.lmbda = lmbda
        self.k_t = k_t
        self.gamma = gamma
        self.norm = norm

        self.discriminator = build_autoencoder()
        optimiser = opt.adam(lr=0.002)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimiser,
                                   metrics=['accuracy'])
        self.discriminator.trainable = False

        self.generator = build_autoencoder()

        # Connect the generator to the discriminator
        gan_input = Input(shape=params.NOISE_SHAPE)
        generated_img = self.generator(gan_input)
        discrim_out = self.discriminator(generated_img)

        # Build and compile the full GAN
        self.combined_model = Model(gan_input, discrim_out)
        optimiser = opt.adam(lr=0.002)
        self.combined_model.compile(loss='binary_crossentropy',
                                    optimizer=optimiser,
                                    metrics=['accuracy'])

    def discriminator_loss(self, y_true, y_pred):
        """
        Custom loss function for Keras. This function is a function of the reconstruction losses
        of both the discriminator and generator.

        The reconstruction loss of the discriminator measures how well it is able to reconstruct
        images of hanliang when passed images of hanliang.

        The reconstruction loss of the generator measures how well it is able to generate an image
        from random noise to minimise the reconstruction loss when this generated image is passed
        to the discriminator.

        The discriminator has two competing objectives. The first is to learn how to encode pictures
        of hanliang, the second is to learn how not to encode pictures generated by the generator.
        In other words, on the one hand it can learn features of hanliang, on the other hand it
        can learn not to encode features in generated images.

        Working towards the first objective creates more accurate images, and working towards the second
        objective creates more diverse images. The tradeoff can be controlled by the gamma parameter.

        See the blog post link above for the full details.

        L_D = discrim_loss - k_t*gen_loss
        L_G = gen_loss

        :param y_true: The image passed to the autoencoder
        :param y_pred: The image reconstructed by the autoencoder
        :return: A float
        """

        # This computes the difference in pixel values between the image passed in and the image
        # reconstructed. The L1 or L2 norm can be applied here ( ie |x-y| or (x-y)^2 )
        discrim_loss_per_pix = (y_true - y_pred).__abs__().__pow__(self.norm)
        discrim_loss = discrim_loss_per_pix.sum()
        gen_loss = self.generator_loss(0, 0)
        loss = discrim_loss - self.k_t*gen_loss
        self.update_kt(discrim_loss, gen_loss)

        return loss

    def generator_loss(self, y_true, y_pred):
        """
        The loss function for the generator. This computes a simple sum of differences
        between real and reconstructed images.

        Alternative: When training the generator, pass the input noise to y_true also,
        then use it here. It will obfuscate the code more though...

        :param y_true: The noise passed to the generator. Not used.
        :param y_pred: The image reconstructed by discriminator. Not used.
        Keras requires these two arguments... so I've included them.
        :return: A float
        """

        noise = np.random.uniform(-1, 1, ((1,) + params.IMG_SHAPE))
        gen_fake_img = self.generator.predict(noise)
        reconstr_fake_img = self.discriminator.predict(gen_fake_img)

        gen_loss_per_pix = (gen_fake_img - reconstr_fake_img).__abs__().__pow__(self.norm)
        gen_loss = gen_loss_per_pix.sum()

        return gen_loss

    def update_kt(self, discrim_loss, gen_loss):
        """
        Update rule for k_t.

        k_t+1 = k_t + lmbda*(gamma*discrim_loss - gen_loss)

        In a perfect world, gamma*discrim_loss - gen_loss == 0. This represents the stable point,
        which the adaptive parameter k_t strives to reach. Let me attempt an intuitive explanation
        to test my understanding of this. To recap, the losses for The discriminator and generator
        are:

        L_D = discrim_loss - k_t*gen_loss
        L_G = gen_loss

        If:

        (1) gen_loss << discrim_loss: generator is getting too good at producing fake images, or
            discriminator is getting lousy at its reconstruction task.
            k_t increases, incentive for discriminator to get better at its discrimination task
            increases (incentive for discriminator to increase gen_loss)
        (2) gen_loss >> discrim_loss: discriminator is getting too good at its encoding task, or
            generator is too lousy at producing fake images.
            k_t decreases, causing a penalty on L_D which helps to prevent it from getting lower.

        :param discrim_loss:
        :param gen_loss:
        :return: Nothing
        """

        self.k_t = self.k_t + self.lmbda*(self.gamma*discrim_loss - gen_loss)


def build_autoencoder():
    """
    Builds an autoencoder which attempts to learn common features in pictures of hanliang.

    :return: A keras Model
    """

    # model = models.Sequential()
    # model.add(layers.Convolution2D(32, (5, 5), activation="relu", input_shape=params.IMG_SHAPE))
    # model.add(layers.MaxPooling2D(pool_size=(5, 5)))
    # model.add(layers.Convolution2D(32, (5, 5), activation="relu")
    # model.add(layers.MaxPooling2D(pool_size=(5, 5)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(1, activation="sigmoid"))
    padding_size = (1, 1)
    kernel_size = (3, 3)

    model = models.Sequential()
    # Start of encoder
    # Shape is still the original image shape
    model.add(layers.ZeroPadding2D(padding=padding_size, input_shape=params.IMG_SHAPE))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))

    # Shape now has half the width and height of the original
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))

    # Shape is now a quarter the width and height of the original
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.Dense(128))

    # Start of decoder
    model.add(layers.Dense(128))
    # Shape is a quarter the width and height of the original
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))

    # Shape is now half the width and height of the original
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))

    # Shape is now the original size
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(64, kernel_size, activation="elu"))
    model.add(layers.ZeroPadding2D(padding=padding_size))
    model.add(layers.Convolution2D(3, kernel_size, activation="elu"))

    return model


# def build_generator():
#     """
#     Builds the generator model, which takes as input random noise and attempts to form an
#     image that would be classified by the discriminator as a han liang.
#
#     :return: A keras Model
#     """
#
#     model = models.Sequential()
#     model.add(layers.Dense(128, input_shape=params.NOISE_SHAPE))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.BatchNormalization(momentum=0.8))
#     model.add(layers.Dense(256))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.BatchNormalization(momentum=0.8))
#     model.add(layers.Dense(256))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.BatchNormalization(momentum=0.8))
#     model.add(layers.Dense(np.prod(params.IMG_SHAPE), activation='tanh'))
#     model.add(layers.Reshape(params.IMG_SHAPE))
#
#     # optimiser = opt.adam(lr=0.002)
#     # model.compile(loss='binary_crossentropy',
#     #               optimizer=optimiser,
#     #               metrics=['accuracy'])
#     return model
