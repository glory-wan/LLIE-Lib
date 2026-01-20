import cv2
import numpy as np

from .AbsMethod import AbsMethod


class MathImage(AbsMethod):
    def __init__(self, param=None, pipeline=None):
        super().__init__(param, pipeline)
        self.gamma = None

    def check_way(self):
        pass

    def gamma_correction(self, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(self.pipeline, table)

    def logarithmic_transformation(self):
        c = 255 / np.log(1 + np.max(self.pipeline))
        log_image = c * (np.log(self.pipeline + 1))
        log_image = np.array(log_image, dtype=np.uint8)

        return log_image
