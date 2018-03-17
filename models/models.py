"""
Builds the discriminator and the generator.
"""
import keras.models as models
import keras.layers as layers
import models.parameters as params


def build_discriminator():
    """
    Builds the model which classifies an image as a picture of hanliang or not.

    :return: A keras Model
    """

    input_shape = (params.WIDTH, params.HEIGHT, params.CHANNELS)

    model = models.Sequential()
    model.add(layers.Convolution2D(32, (10, 10), activation="LeakyRelu"))
    model.add(layers.MaxPooling2D(pool_size=(10, 10)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(2, activation="sigmoid"))

    return model
