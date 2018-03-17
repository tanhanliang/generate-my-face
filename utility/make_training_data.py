"""
Contains functions to build training data for the discriminator.
"""
import utility.preprocessing as pre
import models.parameters as params
import numpy as np
import scipy.misc
import os


def get_training_data():
    """
    Processes all files in the folder specified by path with the .jpg extension. For
    each image in the folder, it will be resized, then
    the input tensor will be built from this.

    :return: A NumPy ndarray with dimensions (training examples, height, width, 3)
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
    return np.reshape(data, newshape=shape)
