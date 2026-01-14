import os
import argparse
import cv2
import math
import numpy as np
from PIL import Image

from .AbsMethod import AbsMethod

"""
    Here is a simple reproduction of the method from Kaiming He's 2009 CVPR paper,
    titled "Single Image Haze Removal Using Dark Channel Prior"
    The paper can be found at this website: https://ieeexplore.ieee.org/document/5206515
        
"""


class DarkChannel(AbsMethod):
    def __init__(self, param=None, size=15):
        super().__init__(param)
        """
            This is the constructor for the DarkChannel class, which implements the Dark Channel Prior method 
            for single image haze removal.

            Args:
                sz (int): The size of the structuring element used in the morphological erosion process 
                          for the dark channel computation.
                          This determines the size of the window over which the minimum filter is applied. 
        """
        self.sz = size
        self.check_way()

    def check_way(self):
        if self.param is not None:
            try:
                self.sz = self.param.size
            except AttributeError:
                self.set_sz()
        else:
            self.set_sz()

    def set_sz(self):
        if self.sz is None:
            self.sz = 15
            print("No 'size' was input, it was set to default=15")

    def dark_channel(self, im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.sz, self.sz))
        dark = cv2.erode(dc, kernel)
        return dark

    def atm_light(self, im, dark):
        [h, w] = im.shape[:2]
        imsz = h * w
        numpx = int(max(math.floor(imsz / 1000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)

        indices = darkvec.argsort()
        indices = indices[imsz - numpx::]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def transmission_estimate(self, im, A):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)

        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]

        transmission = 1 - omega * self.dark_channel(im3)
        return transmission

    @staticmethod
    def guided_filter(im, p, r, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * im + mean_b
        return q

    def transmission_refine(self, im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255
        r = 60  # default 60
        eps = 0.0001
        t = self.guided_filter(gray, et, r, eps)

        return t

    @staticmethod
    def recover(im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)

        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

        return res

    def run(self, image):
        rever_img = 255 - image
        rever_img_bn = rever_img.astype('float64') / 255
        dark = self.dark_channel(rever_img_bn)
        A = self.atm_light(rever_img_bn, dark)
        te = self.transmission_estimate(rever_img_bn, A)
        t = self.transmission_refine(image, te)
        J = self.recover(rever_img_bn, t, A, 0.1)

        rever_res_img = (1 - J) * 255

        return rever_res_img

