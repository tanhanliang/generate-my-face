"""
Contains functions to process the input data, so that they will be ready for input to the model.
"""

from PIL import Image
import models.parameters as params


def resize_and_save_image(name):
    """
    Iterates over all the images in the images/ directory, and resizes them to a given size.
    The resulting images are saved to the processed-images directory.

    :param name: The name of the image. A String.
    :return: Nothing.
    """

    img = Image.open("images/" + name)
    img = img.resize((params.WIDTH, params.HEIGHT), Image.LANCZOS)
    img.save('processed-images/' + name)
