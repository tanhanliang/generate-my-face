"""
Builds the discriminator and the generator.
"""
import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
import models.parameters as params


def build_discriminator():
    """
    Builds the model which classifies an image as a picture of hanliang or not.

    :return: A keras Model
    """

    shape = (params.WIDTH, params.HEIGHT, params.CHANNELS)

    # model = models.Sequential()
    # model.add(layers.Convolution2D(32, (5, 5), activation="relu", input_shape=shape))
    # model.add(layers.MaxPooling2D(pool_size=(5, 5)))
    # model.add(layers.Convolution2D(32, (5, 5), activation="relu", input_shape=shape))
    # model.add(layers.MaxPooling2D(pool_size=(5, 5)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(1, activation="sigmoid"))
    model = models.Sequential()

    model.add(layers.Flatten(input_shape=shape))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
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
