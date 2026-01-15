import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from LLIELib.deepLearning.dataset.imageReader import ImageReader


def get_img_from_floder(folder_path):
    image_files = []
    for root, _, files in os.walk(folder_path):
        current_dir_name = os.path.basename(root)
        for f in tqdm(files, desc=f'Reading images in {current_dir_name}', unit='image'):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'heic')):
                image_files.append(os.path.join(root, f))

    image_files.sort()
    return image_files


def single_data_loader(image_path, transform):
    imReader = ImageReader(return_type='PIL')
    img = imReader(image_path)
    transformer = transforms.Compose([
        transform,
    ])

    imTensor = transformer(img).unsqueeze(0)
    image_name = os.path.basename(image_path)
    basename = os.path.splitext(image_name)   # eg. sample1.jpg
    return imTensor, basename


class PredictDataSet(Dataset):
    def __init__(self, input_dir=None, transform=None, size=(512, 512)):
        self.input_dir = input_dir
        self.image_files = get_img_from_floder(self.input_dir)
        self.imReader = ImageReader(return_type='PIL')

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image_name = os.path.basename(image_path)
        basename = os.path.splitext(image_name)   # eg. sample1.jpg

        img = self.imReader(image_path)
        imgTensor = self.transform(img)

        return imgTensor, basename
