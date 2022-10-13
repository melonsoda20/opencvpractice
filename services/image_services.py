# file import
from .constants.constants import get_image_related_constants


# package import
import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt


IMAGE_CONSTANTS = get_image_related_constants()


def load_img(
    filepath: str,
    desired_width: int = IMAGE_CONSTANTS.IMAGE_WINDOW_DEFAULT_WIDTH,
    desired_height: int = IMAGE_CONSTANTS.IMAGE_WINDOW_DEFAULT_HEIGHT
):
    img = cv.imread(filepath, cv.WINDOW_NORMAL)
    resized_img = cv.resize(img, (desired_width, desired_height))

    while True:
        cv.imshow('Test', resized_img)

        if cv.waitKey(1) & 0xff == 27:
            break
    cv.destroyAllWindows()
