"""
Contains functions to build training data for the discriminator.
"""
import utility.preprocessing as pre
import models.parameters as params
import numpy as np
import scipy.misc
import os
from keras.utils import to_categorical


def get_training_data():
    """
    Processes all files in the folder specified by path with the .jpg extension. For
    each image in the folder, it will be resized, then
    the input tensor will be built from this.

    :return: A tuple of (ndarray, ndarray).
    The first argument has shape (training examples, width, height, channels)
    The second argument has shape (training_examples, number of classes)
    Number of classes should be 2.
    """
    data = []

    for filename in os.listdir("images/"):
        if filename.endswith(".jpg"):
            pre.resize_and_save_image(filename)

    for filename in os.listdir("processed-images/"):
        if filename.endswith(".jpg"):
            img = scipy.misc.imread('processed-images/' + filename, flatten=False, mode='RGB')
            data.append(img)

    shape = (len(data), params.WIDTH, params.HEIGHT, 3)
    data = np.reshape(data, newshape=shape)

    y_vals = np.zeros(len(data))
    y_vals = to_categorical(y_vals, params.CLASS_COUNT)

    return normalise_data(data), y_vals


def normalise_data(data):
    """
    Normalises the data to the range -1, 1. RGB values are in the range 0 to 256.

    :param data: A numpy ndarray
    :return: A numpy ndarray with same dimensions as the input
    """

    return (data-127.5)/127.5
