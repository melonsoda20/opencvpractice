from services.image_services import (
    get_image_data,
    get_opencv_class,
    get_template_matching_results,
    display_image_plots
)
from services.constants.constants import (
    TemplateMatchingConstants
)
from services.models.models import (
    PlotImagesParams
)

opencv_class = get_opencv_class()

# region Template Matching
full_image = get_image_data(
    './images/template_matching/sammy.jpg',
    opencv_class.COLOR_RGB2BGR
)

image_to_be_compared = get_image_data(
    './images/template_matching/sammy_face.jpg',
    opencv_class.COLOR_RGB2BGR
)

selected_template_matching_method = TemplateMatchingConstants.CCORR

template_matching_result = get_template_matching_results(
    full_image,
    image_to_be_compared,
    selected_template_matching_method
)

plots_params = [PlotImagesParams(
        131,
        template_matching_result,
        'Template Matching Results',
        selected_template_matching_method
    ), PlotImagesParams(
        132,
        full_image,
        'Original Image',
        ''
    ),
    PlotImagesParams(
        133,
        image_to_be_compared,
        'Image To Be Compared',
        ''
    )
]
display_image_plots(plots_params)
# endregion
