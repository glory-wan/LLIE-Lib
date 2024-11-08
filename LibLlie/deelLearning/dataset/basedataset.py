import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from LibLlie.deelLearning.dataset.imageReader import ImageReader


def get_img_from_floder(folder_path):
    image_files = []
    for root, _, files in os.walk(folder_path):
        current_dir_name = os.path.basename(root)
        for f in tqdm(files, desc=f'Reading images in {current_dir_name}', unit='image'):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'heic')):
                image_files.append(os.path.join(root, f))

    image_files.sort()
    return image_files


class baseDataSet(Dataset):
    def __init__(self, input_dir=None, transform=None):
        self.input_dir = input_dir
        self.image_files = get_img_from_floder(self.input_dir)
        self.imReader = ImageReader(return_type='PIL')

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image_name = os.path.basename(image_path)
        basename = os.path.splitext(image_name)[0]

        img = self.imReader(image_path)
        imgTensor = self.transform(img)

        return imgTensor, basename
