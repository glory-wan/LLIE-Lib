import os
import time

import torch
import sys
import datetime
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path

from LLIELib.deepLearning.config import *

project_path = Path(__file__).parent.parent.parent
results_path = os.path.join(project_path, 'results')


def log_info_env():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    torch_version = torch.__version__
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 2  # covert to MiB
            print(f"{current_date} Python-{python_version} torch-{torch_version}+cu{cuda_version} "
                  f"CUDA:{i} ({gpu_name}, {int(gpu_memory)}MiB)")
    else:
        print(f"{current_date} Python-{python_version} torch-{torch_version} (No CUDA available)")


def saveImg(tensor, output_dir, name, save_format='.png'):
    name = f'{name}{save_format}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, name)

    image_numpy = tensor[0].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    img = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))

    img.save(save_path)


def showImg(tensor, name):
    image_numpy = tensor[0].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.show(title=name)
    # time.sleep(10)
    im.close()
