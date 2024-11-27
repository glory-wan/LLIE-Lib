import os
import time

import torch
import sys
import datetime
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from LibLlie.deelLearning.config import models, transforms, parameters_ta, model_paths
from LibLlie.deelLearning.dataset.basedataset import baseDataSet, single_data_loader


# print some env info
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
                  f"CUDA:{i + 1} ({gpu_name}, {int(gpu_memory)}MiB)")
    else:
        print(f"{current_date} Python-{python_version} torch-{torch_version} (No CUDA available)")


def save_image(tensor, output_dir, name, save_format='png'):
    name = f'{name}_{int(time.time())}.{save_format}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, name)

    image_numpy = tensor[0].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))

    im.save(save_path, save_format)


def scriptDL(
        model=None,
        input=None,
        output_dir=None,
        save_format='png',

        gpu=0,
        batch_size=1,
        output_height=512,
):
    # env info
    output_format = model
    log_info_env()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n Now, you are using '{device}' to run model!\n")

    # load the model
    model_path = model_paths[model]
    model = models[model]().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    try:
        transform = transforms[output_format]()
    except KeyError:
        transform = transforms['default']()

    if os.path.isdir(input):
        # load test data
        testDataset = baseDataSet(input_dir=input, transform=transform, size=output_height)
        testImages = DataLoader(testDataset, batch_size=batch_size, pin_memory=False, num_workers=os.cpu_count())

        # process images
        model.eval()
        for imTensor, name in tqdm(testImages, desc=f'using {os.path.basename(model_path)} to infer', unit='image'):
            imTensor = imTensor.to(device)
            output, _ = model(imTensor)

            save_image(output, output_dir, name=name[0], save_format=save_format)

    elif os.path.isfile(input):
        # load test data
        imTensor, name = single_data_loader(input, transform, size=output_height)
        # process images
        model.eval()
        imTensor = imTensor.to(device)
        output, _ = model(imTensor)

        save_image(output, output_dir, name=name, save_format=save_format)


def command_DL():
    pta = parameters_ta()
    scriptDL(
        model=pta.model,
        model_path=pta.model_path,
        input=pta.input_dir,
        output_dir=pta.output_dir,
        save_format=pta.save_format,
        gpu=pta.gpu,

        batch_size=pta.batch_size,
        output_height=pta.output_height,
    )
