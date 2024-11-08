import argparse

from LibLlie.deelLearning.model.SCI import Finetunemodel
from LibLlie.deelLearning.model.Zero_DCE import enhance_net_nopool
from LibLlie.deelLearning.dataset.preprocessor import defaultTrans, Zero_DCE_Trans

models = {
    'SCI-easy': Finetunemodel,
    'SCI-medium': Finetunemodel,
    'SCI-difficult': Finetunemodel,
    'Zero-DCE': enhance_net_nopool,
}

transforms = {
    'default': defaultTrans,  # This transforms usually used for SCI model
    'Zero-DCE': Zero_DCE_Trans,
}


def parameters_ta():
    parser = argparse.ArgumentParser(description='Configuration parameters of deeplearning algorithms')

    # model parameters
    parser.add_argument('--model', type=str, required=True, help='the deeplearning algorithm', default='Zero-DCE')
    parser.add_argument('--model_path', type=str, required=True,
                        help='the weights which will be load, saved in \'LibLlie/models\'',
                        default='LibLlie/models/Zero-DCE/Zero-DCE.pth')

    # data parameters
    parser.add_argument('--input_dir', type=str, help='the format of the image which will be handle (required)',
                        default='assets/DL_test/input')
    parser.add_argument('--output_dir', type=str, help='the directory of the image which will be saved (required)',
                        default='assets/DL_test/output')
    parser.add_argument('--save_format', type=str, help='the format of the image which will be saved', default='jpg')

    # eval parameters
    parser.add_argument('--batch_size', type=int, help='the number of image of one batch', default=1)
    parser.add_argument('--output_height', type=int, help='the height of the output image', default=512)

    return parser.parse_args()
