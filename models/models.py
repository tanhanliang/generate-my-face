"""
Builds the discriminator and the generator.
"""
import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
import models.parameters as params
import numpy as np


def build_discriminator():
    """
    Builds the model which classifies an image as a picture of han liang or not.

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
    model = models.Sequential()
    # model.add(layers.Flatten(input_shape=params.IMG_SHAPE))
    # model.add(layers.Dense(32))
    model.add(layers.Convolution2D(8, (5, 5), activation="relu", input_shape=params.IMG_SHAPE))
    model.add(layers.MaxPooling2D(pool_size=(5, 5)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(32))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(1, activation='sigmoid'))

    optimiser = opt.adam(lr=0.002)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model


def build_generator():
    """
    Builds the generator model, which takes as input random noise and attempts to form an
    image that would be classified by the discriminator as a han liang.

    :return: A keras Model
    """

    model = models.Sequential()
    model.add(layers.Dense(128, input_shape=params.NOISE_SHAPE))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(params.IMG_SHAPE), activation='tanh'))
    model.add(layers.Reshape(params.IMG_SHAPE))

    optimiser = opt.adam(lr=0.002)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model
