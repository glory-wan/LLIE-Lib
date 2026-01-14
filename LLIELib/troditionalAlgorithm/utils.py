import cv2
import os
from pathlib import Path

from ..utils import ReadImage, show_img, save_img, get_scaled_size
from LLIELib.troditionalAlgorithm.methods.HEMethod import HeImage
from LLIELib.troditionalAlgorithm.methods.DarkChannel import DarkChannel
from LLIELib.troditionalAlgorithm.config import parameters_ta
from LLIELib.troditionalAlgorithm.processPipeline import process_pipeline


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


class TAProcessor:
    def __init__(self,
                 img_path=None,
                 algorithm=None,
                 color_space=None,
                 showimg=False,
                 saveimg=True,
                 name_Suffix='',
                 width=None,
                 height=None,
                 format=None,
                 save_dir=results_path,
                 clipLimit=None,
                 gridSize=None,
                 iteration=None,
                 dcpSize=None,
                 ):
        self.img_path = img_path
        self.algorithm = algorithm
        self.color_space = color_space
        self.showimg = showimg
        self.saveimg = saveimg
        self.name_Suffix = name_Suffix
        self.width = width
        self.height = height
        self.format = format
        self.iteration = iteration
        self.dcpSize = dcpSize
        self.save_dir = save_dir
        self.clipLimit = clipLimit
        self.gridSize = gridSize

    def processor(self, img_path):
        self.img_path = img_path
        reader = ReadImage(self.img_path)
        image = reader.img

        file_name = os.path.basename(self.img_path)  # example.jpg
        name = f'{os.path.splitext(file_name)[0]}{self.name_Suffix}'
        if self.format is None:
            self.format = os.path.splitext(file_name)[1]  # .png

        if self.width is None or self.height is None:
            height, width = image.shape[:2]
            if self.showimg:
                show_h, show_w = get_scaled_size(height, width)
            else:
                show_h, show_w = height, width
        else:
            show_h, show_w = self.height, self.width

        pipeliner = PipelineImage(img=image)
        processor = HeImage(
            pipeline=pipeliner,
            clipLimit=self.clipLimit,
            gridSize=self.gridSize,
            iteration=self.iteration
        )

        if self.algorithm == 'DCP':
            DCP = DarkChannel(size=self.dcpSize)
        else:
            DCP = None

        al_cs = AlgorithmCs(pipeline=pipeliner, process=processor)
        he_algorithms = al_cs.he_algorithm()
        color_spaces = al_cs.color_space

        image = process_pipeline(
            method=self.algorithm,
            cs=self.color_space,
            algorithms=he_algorithms,
            color_space=color_spaces,
            process=processor,
            pipeline=pipeliner,
            DCP=DCP,
        )

        if self.showimg:
            show_img(
                image,
                name=name,
                width=show_w,
                height=show_h
            )
        if self.saveimg:
            save_img(
                image,
                name=name,
                format=self.format,
                directory=self.save_dir,
            )

        return image

    def command(self):
        pta = parameters_ta()
        self.img_path = pta.img
        self.algorithm = pta.algorithm
        self.color_space = pta.cs
        self.showimg = pta.showing
        self.saveimg = pta.save
        self.name_Suffix = pta.name_Suffix
        self.width = pta.width
        self.height = pta.height
        self.format = pta.format
        self.iteration = pta.iteration
        self.dcpSize = pta.dcpSize
        self.save_dir = pta.save_dir
        self.clipLimit = pta.clipLimit
        self.gridSize = pta.gridSize

        image = self.processor(img_path=self.img_path)

        return image

