from services.image_services import (
    display_img,
    get_image_data,
    get_opencv_class,
    get_template_matching_results,
    display_image_plots,
    get_harris_corner_detection,
    detect_image_using_cascade
)
from services.constants.constants import (
    TemplateMatchingConstants
)
from services.models.models import (
    PlotImagesParams,
    GetHarrisCornerDetectionParams,
    DetectImageUsingCascadeClassfierParams
)

opencv_class = get_opencv_class()

# region Template Matching
# full_image = get_image_data(
#     './images/template_matching/sammy.jpg',
#     opencv_class.COLOR_RGB2BGR
# )

# image_to_be_compared = get_image_data(
#     './images/template_matching/sammy_face.jpg',
#     opencv_class.COLOR_RGB2BGR
# )

# selected_template_matching_method = TemplateMatchingConstants.CCORR

# template_matching_result = get_template_matching_results(
#     full_image,
#     image_to_be_compared,
#     selected_template_matching_method
# )

# plots_params = [PlotImagesParams(
#         131,
#         template_matching_result,
#         'Template Matching Results',
#         selected_template_matching_method
#     ), PlotImagesParams(
#         132,
#         full_image,
#         'Original Image',
#         ''
#     ),
#     PlotImagesParams(
#         133,
#         image_to_be_compared,
#         'Image To Be Compared',
#         ''
#     )
# ]
# display_image_plots(plots_params)
# endregion

# region Corner Detection
# flat_chess_image = get_image_data(
#     './images/corner_detection/flat_chessboard.png',
#     opencv_class.COLOR_BGR2RGB
# )

# gray_flat_chess_image = get_image_data(
#     './images/corner_detection/flat_chessboard.png',
#     opencv_class.COLOR_BGR2GRAY
# )

# real_chess_image = get_image_data(
#     './images/corner_detection/real_chessboard.jpg',
#     opencv_class.COLOR_BGR2RGB
# )

# gray_real_chess_image = get_image_data(
#     './images/corner_detection/real_chessboard.jpg',
#     opencv_class.COLOR_BGR2GRAY
# )

# harris_corner_res = get_harris_corner_detection(
#     GetHarrisCornerDetectionParams(
#         real_chess_image,
#         2,
#         3,
#         0.04
#     )
# )

# plots_params = [PlotImagesParams(
#         121,
#         real_chess_image,
#         'Original Image',
#         ''
#     ), PlotImagesParams(
#         122,
#         harris_corner_res,
#         'After Harris Corner',
#         '',
#         ''
#     )
# ]
# display_image_plots(plots_params)
# endregion

# region detection for vox project
vehicle_image = get_image_data(
    './images/yeh-che-wei-Q78DeDcXUzQ-unsplash.jpg',
    opencv_class.COLOR_RGB2BGR
)

params = DetectImageUsingCascadeClassfierParams(vehicle_image)
image_res = detect_image_using_cascade(params)

plots_params = [PlotImagesParams(
        121,
        vehicle_image,
        'Original Image',
        ''
    ), PlotImagesParams(
        122,
        opencv_class.cvtColor(image_res, opencv_class.COLOR_BGR2RGB),
        'After Detection',
        ''
    )
]
display_image_plots(plots_params)
# endregion
