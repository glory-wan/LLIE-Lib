import torch
import datetime
import sys
import numpy as np
from PIL import Image


def log_info_env():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    torch_version = torch.__version__
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**2  # covert to MiB
            print(f"{current_date} Python-{python_version} torch-{torch_version}+cu{cuda_version} "
                  f"CUDA:{i+1} ({gpu_name}, {int(gpu_memory)}MiB)")
    else:
        print(f"{current_date} Python-{python_version} torch-{torch_version} (No CUDA available)")


def save_image(tensor, path, _format='png'):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, _format)

