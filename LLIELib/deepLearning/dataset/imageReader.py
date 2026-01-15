import os
import numpy as np
from PIL import Image
# import pillow_heif
import cv2


class ImageReader:
    def __init__(self, img_path=None, return_type='PIL', gray=False):
        """
        Initialize the ImageReader.

        Args:
            img_path (str): Path to the image file.
            return_type (str): Type of the returned image. Options are:
                               'PIL' (default) - returns a PIL Image object.
                               'numpy' - returns a numpy ndarray.
            gray (bool): Whether to return the image in grayscale.
        """
        self.img_path = img_path
        self.return_gray = gray
        self.return_type = return_type.lower()

        if self.return_type not in ['pil', 'numpy']:
            raise ValueError(f"Unsupported return_type: {self.return_type}. Use 'PIL' or 'numpy'.")

    def read_heic(self):
        """Reads a HEIC image and returns it as a PIL Image."""
        try:
            heif_file = pillow_heif.read_heif(self.img_path)
            img = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            return img
        except Exception as e:
            raise IOError(f"Unable to read HEIC image: {self.img_path}") from e

    def read_image(self):
        """Reads an image and returns it in the format specified by return_type."""
        try:
            _, ext = os.path.splitext(self.img_path)
            # if ext.lower() == '.heic':
            #     img = self.read_heic()
            # else:
            #     img = Image.open(self.img_path)
            img = Image.open(self.img_path).convert('RGB')

            if self.return_gray:
                img = img.convert('L')

            if self.return_type == 'numpy':
                img = self.convert_to_numpy(img)

                if self.return_gray and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            return img

        except (FileNotFoundError, IOError) as e:
            raise FileNotFoundError(f"Image not found or unable to read: {self.img_path}") from e
        except ValueError as e:
            raise ValueError(f"Unsupported file type: {self.img_path}") from e

    @staticmethod
    def convert_to_numpy(img):
        """Converts a PIL image to a numpy ndarray."""
        return np.array(img)

    def __call__(self, img_path):
        self.img_path = img_path
        return self.read_image()


# if __name__ == "__main__":
#     reader = ImageReader(return_type='numpy')
#
#     img = reader(r"D:\Code\pycode\Data_All\LLIE_paired\normal\demo\000000000285.jpg")
#
#     print(type(img))
#     print(img)


