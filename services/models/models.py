from typing import Tuple

import cv2 as cv


class DrawingPosition:
    def __init__(
        self,
        TopLeft: Tuple[int, int] = (0, 0),
        TopRight: Tuple[int, int] = (0, 0),
        BottomLeft: Tuple[int, int] = (0, 0),
        BottomRight: Tuple[int, int] = (0, 0)
    ) -> None:
        self.TopLeft = TopLeft
        self.TopRight = TopRight
        self.BottomLeft = BottomLeft
        self.BottomRight = BottomRight


class DrawRectangle:
    def __init__(
        self,
        ImageToBeDrawn: cv.Mat,
        DrawPositions: DrawingPosition,
        Color: Tuple[int, int, int],
        Thickness: int
    ) -> None:
        self.ImageToBeDrawn = ImageToBeDrawn
        self.DrawPositions = DrawPositions
        self.Color = Color
        self.Thickness = Thickness


class PlotImagesParams:
    def __init__(
        self,
        Subplot: int,
        Image: cv.Mat,
        Title: str,
        Suptitle: str,
        Cmap: str = ''
    ) -> None:
        self.Subplot = Subplot
        self.Image = Image
        self.Title = Title
        self.Suptitle = Suptitle
        self.Cmap = Cmap


class GetHarrisCornerDetectionParams:
    def __init__(
        self,
        Image: cv.Mat,
        BlockSize: int,
        KSize: int,
        K: float
    ) -> None:
        self.Image = Image
        self.BlockSize = BlockSize
        self.KSize = KSize
        self.K = K


class DetectImageUsingCascadeClassfierParams:
    def __init__(
        self,
        ImageData: cv.Mat
    ) -> None:
        self.ImageData = ImageData


class CascadeClassifiersData:
    def __init__(
        self,
        CarClassifier: cv.CascadeClassifier,
        HumanClassifier: cv.CascadeClassifier
    ) -> None:
        self.CarClassifier = CarClassifier
        self.HumanClassifier = HumanClassifier
