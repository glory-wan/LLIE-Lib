import cv2
import numpy as np
import os
import base64
import requests
from io import BytesIO
from pathlib import Path



class ReadImage:
    """
       A class for reading images from local files, URLs, or byte streams.

       Args:
           image_source (str or bytes): Path to the local image file, URL of the image,
                                                                    or a byte stream representing the image.
           timeout (int, optional): Timeout in seconds for fetching the image from URL (default: 10).

       Attributes:
           image_type (str): The type of the image source ('local', 'url', or 'bytes').
           img (numpy.ndarray): The image data as a NumPy array.
    """

    def __init__(self, image_source=None, timeout=10):
        self.image_source = image_source
        self.timeout = timeout
        self.image_type = self.check_type()
        self.img = self.read_image()

    def check_type(self):
        if isinstance(self.image_source, bytes):
            return "bytes"
        elif self.image_source and os.path.isfile(self.image_source):
            return "local"
        elif self.image_source:
            return "url"
        else:
            return None

    def read_image(self):
        """
        Reads the image from the specified source and returns it as a NumPy array.

        Returns:
            numpy.ndarray: The image data as a NumPy array, or None if the image could not be read.
        """
        try:
            if self.image_type == "local":
                return cv2.imread(self.image_source, cv2.IMREAD_COLOR)
            elif self.image_type == "url":
                try:
                    response = requests.get(self.image_source, timeout=self.timeout)
                    response.raise_for_status()
                    if response.headers.get("Content-Type", "").startswith("image/"):
                        image_data = BytesIO(response.content)
                        return cv2.imdecode(np.frombuffer(image_data.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    else:
                        print(f"Error: Content-Type '{response.headers['Content-Type']}' is not an image.")
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching image: {e}")
            elif self.image_type == "bytes":
                converter = ConvertFormat(data=self.image_source, convertWay='bytes2img')
                return converter.convert_data()
        except Exception as e:
            print(f"Error reading image: {e}")

            return None


class ConvertFormat:
    def __init__(self, data=None, convertWay=None, ext='jpg'):
        self.data = data
        self.ext = ext
        self.convertWay = convertWay
        self.format = {
            'bytes2img': self.bytes_to_image,
            'img2bytes': self.image_to_bytes,
            'base642img': self.base64_to_img,
            'img2base64': self.img_to_base64
        }

    def convert_data(self):
        if self.data is None:
            raise ValueError("No data stream is input.")

        convertFunction = self.format[self.convertWay]
        convertFunction()

        if self.data is None:
            raise ValueError("Cannot convert format of image.")

        return self.data

    def bytes_to_image(self):
        img_array = np.frombuffer(self.data, np.uint8)
        self.data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return self.data

    def image_to_bytes(self):
        # _, encoded_img = cv2.imencode(f'.{self.ext}', self.data)[1].tobytes()
        _, encoded_img = cv2.imencode(f'.{self.ext}', self.data)
        self.data = encoded_img.tobytes()

        return self.data

    def base64_to_img(self):
        image_bytes = base64.b64decode(self.data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        self.data = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        return self.data

    def img_to_base64(self):
        _, encoded_img = cv2.imencode(f'.{self.ext}', self.data)
        base64_bytes = base64.b64encode(encoded_img)
        self.data = base64_bytes.decode('utf-8')

        return self.data


project_path = Path(__file__).parent.parent
results_path = os.path.join(project_path, 'results')


def save_img(img, name='resultImg', format='png', directory=results_path):
    """
        This function is used to save the processed image
    :param img: the image which need to be saved
    :param directory: the directory where the image will be saved
    :param name: the name of processed image
    :param format: the format of processed image
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"{name}.{format}")
    cv2.imwrite(file_path, img)
    print(f"Image has been saved in {file_path}")


def show_img(img, name='resultImg', width=800, height=600):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_scaled_size(height, width, max_size: int = 1024, min_size: int = 512):
    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    if min(new_width, new_height) < min_size:
        scale = min_size / min(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)

    return new_height, new_width