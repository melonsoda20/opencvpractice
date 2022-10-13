class ImageConstants:
    def __init__(
        self,
        IMAGE_WINDOW_DEFAULT_WIDTH: int,
        IMAGE_WINDOW_DEFAULT_HEIGHT: int
    ) -> None:
        self.IMAGE_WINDOW_DEFAULT_WIDTH = IMAGE_WINDOW_DEFAULT_WIDTH
        self.IMAGE_WINDOW_DEFAULT_HEIGHT = IMAGE_WINDOW_DEFAULT_HEIGHT


def get_image_related_constants() -> ImageConstants:
    return ImageConstants(960, 540)
