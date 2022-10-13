import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt


def load_img(filepath: str):
    img = cv.imread(filepath, cv.WINDOW_NORMAL)
    resized_img = cv.resize(img, (960, 540))

    while True:
        cv.imshow('Test', resized_img)

        if cv.waitKey(1) & 0xff == 27:
            break
    cv.destroyAllWindows()


i = load_img('./images/hasbi-kurnia-QYETv_HlkCY-unsplash.jpg')
