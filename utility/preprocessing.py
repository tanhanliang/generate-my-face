"""
Contains functions to process the input data, so that they will be ready for input to the model.
"""

from PIL import Image


def standardise_image_size(path, name, height, width):
    """
    Iterates over all the images in the images/ directory, and resizes them to a given size.
    The resulting images are saved to the processed-images directory.

    :param path: The path to the image. A String.
    :param name: The name of the image. A String.
    :param height: The resulting height in pixels. An integer.
    :param width: The resulting width in pixels. An integer.
    :return: Nothing.
    """

    img = Image.open(path)
    img = img.resize((width, height), Image.LANCZOS)
    img.save('processed-images/' + name)
