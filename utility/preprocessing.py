"""
Contains functions to process the input data, so that they will be ready for input to the model.
"""

from PIL import Image
import models.parameters as params


def resize_image(name, folder):
    """
    Iterates over all the images in the images/ directory, and resizes them to a given size.
    The resulting images are saved to the processed-images directory.

    :param name: The name of the image. A String.
    :param folder: The folder containing the image
    :return: Nothing.
    """

    img = Image.open(folder + name)
    img = img.resize((params.WIDTH, params.HEIGHT), Image.LANCZOS)
    return img
