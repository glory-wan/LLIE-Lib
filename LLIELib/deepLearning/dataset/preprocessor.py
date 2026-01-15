import torch
from torchvision.transforms import v2

from LLIELib.deepLearning.dataset.imageReader import ImageReader


class predict_Trans(v2.Compose):
    def __init__(self,):
        super(predict_Trans, self).__init__([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Resize((height, width)),
        ])



