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
    def __init__(self):
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
