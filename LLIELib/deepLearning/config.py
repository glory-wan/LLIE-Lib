import argparse
import os.path
from email.policy import default

from LLIELib.deepLearning.model import Finetunemodel, Zero_DCE, Zero_DCE_extension

models = {
    'SCI-easy': Finetunemodel,
    'SCI-medium': Finetunemodel,
    'SCI-difficult': Finetunemodel,
    'Zero_DCE': Zero_DCE,
    'Zero_DCE_extension': Zero_DCE_extension
}

# transforms = {
#     'default': defaultTrans,
#     'Zero_DCE': Zero_DCE_Trans,
# }


def parameters_dl():
    parser = argparse.ArgumentParser(description='Configuration parameters of deeplearning algorithms')

    # model parameters
    parser.add_argument('--model', type=str, required=True,help='the deeplearning model',
                        default='Zero_DCE')
    parser.add_argument('--weight', type=str, required=True, help="the deeplearning model's weights",
                        default='./weights/Zero_DCE/Zero_DCE.pth')
    # data parameters
    parser.add_argument('--input', type=str, help='the format of the image which will be handle (required)',
                        default='assets/input.jpg', required=True)
    parser.add_argument('--save_dir', type=str, help='the directory of the image which will be saved (required)',
                        default='results')
    parser.add_argument('--save_format', type=str, help='the format of the image which will be saved')

    # eval parameters
    parser.add_argument('--save_image', type=bool, help='whether the image will be saved.', default=True)
    parser.add_argument('--show_image', type=bool, help='whether the image will be showed.', default=False)

    parser.add_argument('--gpu', type=int, help='gpu device id', default=0)
    parser.add_argument('--batch_size', type=int, help='the number of image of one batch', default=1)
    # parser.add_argument('--width', type=int, default=512, help='the width of the output image')
    # parser.add_argument('--height', type=int, default=512, help='the height of the output image')
    parser.add_argument('--name_Suffix', type=str, default='', help='Suffix to append to output image filenames. ')

    return parser.parse_args()
