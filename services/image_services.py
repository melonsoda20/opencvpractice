# file import
from typing import List
from cv2 import Mat
from .constants.constants import (
    ImageConstants,
    TemplateMatchingConstants
)
from .models.models import (
    DrawingPosition,
    DrawRectangle,
    PlotImagesParams,
    GetHarrisCornerDetectionParams,
    DetectImageUsingCascadeClassfierParams,
    CascadeClassifiersData
)


# package import
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


IMAGE_CONSTANTS = ImageConstants()
TEMPLATE_MATCHING_METHOD_CONSTANTS = TemplateMatchingConstants()


def get_opencv_class():
    return cv


def display_img(
    image_data: Mat,
    desired_width: int = IMAGE_CONSTANTS.IMAGE_WINDOW_DEFAULT_WIDTH,
    desired_height: int = IMAGE_CONSTANTS.IMAGE_WINDOW_DEFAULT_HEIGHT,
    color: int = cv.COLOR_BGR2RGB
):
    resized_img = cv.resize(image_data, (desired_width, desired_height))
    resized_img = cv.cvtColor(resized_img, color)

    while True:
        cv.imshow('Test', resized_img)

        if cv.waitKey(1) & 0xff == 27:
            break
    cv.destroyAllWindows()


def display_image_plots(
    params: List[PlotImagesParams]
):
    for param in params:
        plt.subplot(param.Subplot)
        if param.Cmap != '':
            plt.imshow(param.Image, cmap=param.Cmap)
        else:
            plt.imshow(param.Image)
        plt.title(param.Title)
        plt.suptitle(param.Suptitle)

    plt.show(block=True)


def get_image_data(
    filepath: str,
    color: int = cv.COLOR_BGR2RGB
):
    img = cv.imread(filepath)
    updated_img = cv.cvtColor(img, color)
    return updated_img


def get_template_matching_results(
    full_image: Mat,
    image_to_be_compared: Mat,
    template_matching_method: str
) -> Mat:
    # Create a Copy og the image
    full_image_copy = full_image.copy()

    method = eval(template_matching_method)

    # Template Matching
    res = cv.matchTemplate(
        full_image_copy,
        image_to_be_compared,
        method
    )

    _, _, min_loc, max_loc = cv.minMaxLoc(res)
    drawing_position = DrawingPosition()
    if (method == TEMPLATE_MATCHING_METHOD_CONSTANTS.SQDIFF or
            method == TEMPLATE_MATCHING_METHOD_CONSTANTS.SQDIFF_NORMED):
        drawing_position.TopLeft = min_loc
    else:
        drawing_position.TopLeft = max_loc

    height, width, _ = image_to_be_compared.shape

    drawing_position.BottomRight = (
        drawing_position.TopLeft[0] + width,
        drawing_position.TopLeft[1] + height
    )
    draw_rectangle_params = DrawRectangle(
        full_image_copy,
        drawing_position,
        (255, 0, 0),
        10
    )

    drawn_image = draw_rectangle(draw_rectangle_params)

    return drawn_image


def draw_rectangle(
    params: DrawRectangle
) -> Mat:
    return cv.rectangle(
        params.ImageToBeDrawn,
        params.DrawPositions.TopLeft,
        params.DrawPositions.BottomRight,
        params.Color,
        params.Thickness
    )


def get_harris_corner_detection(
    params: GetHarrisCornerDetectionParams
):
    image_copy = params.Image.copy()
    gray_image = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
    np_data = gray_image.astype(np.float32)
    result = cv.cornerHarris(
        src=np_data,
        blockSize=params.BlockSize,
        ksize=params.KSize,
        k=params.K
    )

    result = cv.dilate(result, None)
    image_copy[result > 0.01*result.max()] = [255, 0, 0]

    return image_copy


def get_cascade_classifier() -> CascadeClassifiersData:
    car_classifier = cv.CascadeClassifier(
        'F:\\Projects\\Python\\OpenCV\\services\\cascade_classifier\\cars.xml'
    )
    human_classifier = cv.CascadeClassifier(
        'F:\\Projects\\Python\\OpenCV\\services\\cascade_classifier\\fullbody.xml'
    )

    return CascadeClassifiersData(car_classifier, human_classifier)


def detect_image_using_cascade(
    params: DetectImageUsingCascadeClassfierParams
) -> Mat:
    cascade_classifier = get_cascade_classifier()
    image_copy = params.ImageData.copy()
    colored_image = cv.cvtColor(image_copy, cv.COLOR_BGR2RGB)
    resized_img = cv.resize(colored_image, (450, 450))
    gray_image = cv.cvtColor(resized_img, cv.COLOR_RGB2GRAY)
    blur_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    dilated_image = cv.dilate(blur_image, np.ones((3, 3)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    closing_image = cv.morphologyEx(dilated_image, cv.MORPH_CLOSE, kernel)

    detected_cars = cascade_classifier.CarClassifier.detectMultiScale(
        closing_image,
        1.1,
        1
    )

    detected_humans = cascade_classifier.HumanClassifier.detectMultiScale(
        closing_image,
        1.1,
        1
    )

    for (x, y, w, h) in detected_cars:
        cv.rectangle(resized_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in detected_humans:
        cv.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('final', resized_img)
    return resized_img
