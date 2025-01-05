import cv2
import os
from pathlib import Path

from ..utils import ReadImage, show_img, save_img
from LibLlie.troditionAlgorithm.methods.heMethod import HeImage
from LibLlie.troditionAlgorithm.methods.DarkChannel import DarkChannel
from LibLlie.troditionAlgorithm.config import parameters_ta
from LibLlie.troditionAlgorithm.processPipeline import process_pipeline


class PipelineImage:
    def __init__(self, img=None):
        self.img = img
        self.pipeline1 = None
        self.pipeline2 = None
        self.pipeline3 = None
        self.pipeline = None

    def check_img(self):
        if self.img is None:
            raise ValueError("Image not loaded properly.")

    # def accept_pipeline(self, pipeline):
    #     """
    #     Receives the modified channel value
    #     if color space is RGB, it's going to be a three-channel array,
    #     otherwise it will be a single-channel array
    #     """
    #     self.pipeline = pipeline

    def merge_pipeline(self, cs):
        if cs == 'rgb':
            self.img = cv2.merge((self.pipeline[0], self.pipeline[1], self.pipeline[2]))
        elif cs == 'hls':
            self.img = cv2.merge((self.pipeline1, self.pipeline, self.pipeline3))
            self.img = cv2.cvtColor(self.img, cv2.COLOR_HLS2BGR)
        elif cs == 'hsv':
            self.img = cv2.merge((self.pipeline1, self.pipeline2, self.pipeline))
            self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
        elif cs == 'lab':
            self.img = cv2.merge((self.pipeline, self.pipeline2, self.pipeline3))
            self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
        elif cs == 'yuv':
            self.img = cv2.merge((self.pipeline, self.pipeline2, self.pipeline3))
            self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
        else:
            self.img = None
            print(f"Unsupported color_space: {cs}. "
                  f"Use 'rgb', 'hsv', 'hls', 'lab', 'yuv'.")

        return self.img

    def rgb(self):
        """
        Returns:
            A tuple containing the three channels: blue channel, green channel, and red channel.
        """
        self.check_img()
        self.pipeline1, self.pipeline2, self.pipeline3 = cv2.split(self.img)
        return self.pipeline1, self.pipeline2, self.pipeline3

    def hsv(self):
        self.check_img()
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.pipeline1, self.pipeline2, self.pipeline3 = cv2.split(hsv_img)
        return self.pipeline3

    def yuv(self):
        self.check_img()
        yuv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        self.pipeline1, self.pipeline2, self.pipeline3 = cv2.split(yuv_img)
        return self.pipeline1

    def hls(self):
        self.check_img()
        hls_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        self.pipeline1, self.pipeline2, self.pipeline3 = cv2.split(hls_img)
        return self.pipeline2

    def lab(self):
        self.check_img()
        lab_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.pipeline1, self.pipeline2, self.pipeline3 = cv2.split(lab_img)
        return self.pipeline1


class AlgorithmCs:
    def __init__(self, process, pipeline):
        self.processor = process
        self.pipeline = pipeline
        self.color_space = {
            'rgb': self.pipeline.rgb,
            'hsv': self.pipeline.hsv,
            'yuv': self.pipeline.yuv,
            'hls': self.pipeline.hls,
            'lab': self.pipeline.lab,
        }

    def he_algorithm(self):
        algorithms = {
            'he': self.processor.he,
            'clahe': self.processor.clahe,
            'rclahe': self.processor.recursive_clahe,
        }

        return algorithms


project_path = Path(__file__).parent.parent.parent
results_path = os.path.join(project_path, 'results')


def script_ta(img_path,
              algorithm,
              color_space,
              showimg=True,
              saveimg=True,
              name='resultImg',
              width=800,
              height=600,
              format='jpg',
              directory=results_path,
              clipLimit=None,
              gridSize=None,
              iteration=None,
              dcpSize=None,
              ):
    reader = ReadImage(img_path)
    image = reader.img

    pipeliner = PipelineImage(img=image)
    processor = HeImage(
        pipeline=pipeliner,
        clipLimit=clipLimit,
        gridSize=gridSize,
        iteration=iteration
    )

    if algorithm == 'DCP':
        DCP = DarkChannel(size=dcpSize)
    else:
        DCP = None

    al_cs = AlgorithmCs(pipeline=pipeliner, process=processor)
    he_algorithms = al_cs.he_algorithm()
    color_spaces = al_cs.color_space

    image = process_pipeline(
        method=algorithm,
        cs=color_space,
        algorithms=he_algorithms,
        color_space=color_spaces,
        process=processor,
        pipeline=pipeliner,
        DCP=DCP,
    )
    if showimg:
        show_img(
            image,
            name=name,
            width=width,
            height=height
        )
    if saveimg:
        save_img(
            image,
            name=name,
            format=format,
            directory=directory
        )

    return image


def command_ta():
    pta = parameters_ta()
    reader = ReadImage(pta.img)
    image = reader.img

    pipeliner = PipelineImage(img=image)

    processor = HeImage(
        param=pta,
        pipeline=pipeliner,
        clipLimit=pta.clipLimit,
        gridSize=pta.gridSize,
        iteration=pta.iteration
    )

    al_cs = AlgorithmCs(process=processor, pipeline=pipeliner)
    algorithms = al_cs.he_algorithm()
    color_spaces = al_cs.color_space

    if pta.method == 'DCP':
        DCP = DarkChannel(size=pta.size)
    else:
        DCP = None

    image = process_pipeline(
        method=pta.method,
        cs=pta.cs,
        algorithms=algorithms,
        color_space=color_spaces,
        process=processor,
        pipeline=pipeliner,
        DCP=DCP,
    )
    if pta.display:
        show_img(
            image,
            name=pta.name,
            width=pta.width,
            height=pta.height
        )

    save_img(
        image,
        directory=pta.save,
        name=pta.name,
        format=pta.format
    )
