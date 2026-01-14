import argparse
import os.path
from pathlib import Path


project_path = Path(__file__).parent.parent.parent
results_path = os.path.join(project_path, 'results')


def parameters_ta():
    parser = argparse.ArgumentParser(description='Configuration parameters of traditional algorithms')

    # the following parameters are necessary
    parser.add_argument('--img', type=str, required=True, help='The path including url and local to an image file')
    parser.add_argument('--algorithm', type=str, required=True, help='The specific kind of Histogram equalization')
    parser.add_argument('--cs', type=str, help='The specific kind of color space')

    # The following parameters are alternative, but they have some impact on processing the image.
    parser.add_argument('--clipLimit', type=float, default=2.0, help='Threshold for contrast limiting.')
    parser.add_argument('--gridSize', type=int, default=8, help='Size of the grid for the histogram equalization.')
    parser.add_argument('--iteration', type=int, default=2, help='The number of recursive calls')

    # The following parameters are optional and do not affect the image processing.
    parser.add_argument('--name_Suffix', type=str, default='', help='The name of the output image file')
    parser.add_argument('--save', default=True, type=bool, help='Whether to save the image or not')
    parser.add_argument('--save_dir', type=str, default=results_path,
                        help='A directory path where the result image will be saved.')
    parser.add_argument('--format', type=str, default='jpg', help='the format of the image which will be saved')
    parser.add_argument('--showing', default=False, help='Select to display the processed image')
    parser.add_argument('--width', type=int, default=800,
                        help='The width of the window in which the picture is displayed')
    parser.add_argument('--height', type=int, default=600,
                        help='The height of the window in which the picture is displayed')

    parser.add_argument('--dcpSize', type=int, default=15, help='The kernel size for DarkChannel extraction')

    return parser.parse_args()



