import os
import cv2
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from models_zoo import get_model, get_model_stat


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(image_path, resize_shape=None):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(resize_shape),
    ])
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0)


def get_feature_map(model, x):
    layers = list(model.children())[:-2]
    feature_extractor = torch.nn.Sequential(*layers)
    start_time = time.time()
    output_tensor = feature_extractor(x)
    inference_time = time.time() - start_time
    inference_time = inference_time * 1e3

    return output_tensor, inference_time


def visualize_feature_map(feature_map, save_path, infer_time):
    feature_map = feature_map.squeeze(0)
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map[0].cpu(), cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'save {os.path.basename(save_path)}, inference_time = {infer_time}(ms)')


if __name__ == '__main__':
    set_seed(142)  # 设置随机种子
    folder_path = r"D:\Code\pycode\Data_All\Database_of_CV\Experiment\analysis\show"
    model_name = 'vit_l_32'  # 替换为其他模型名称
    model = get_model(model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'device is {device}')

    model.to(device)

    stats = {'model': model_name}
    infer_time = 0
    json_path = None
    model_info_path = None
    input_tensor = None
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            basedir = os.path.dirname(image_path)
            basedir = os.path.join(basedir, model_name)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
                print(f"Directory {basedir} created.")
            nameImg = os.path.splitext(os.path.basename(image_path))[0]
            save_name = f"{nameImg}_{model_name}.png"
            save_path = os.path.join(basedir, save_name)
            json_path = os.path.join(basedir, f"{model_name}_stats.json")
            model_info_path = os.path.join(basedir, f"{model_name}_info.txt")

            input_tensor = load_image(image_path)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                feature_map, inference_time = get_feature_map(
                    model=model,
                    x=input_tensor,
                )

            visualize_feature_map(feature_map,
                                  save_path=save_path,
                                  infer_time=inference_time,
                                  )

    # get info of model
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    flops, params = get_model_stat(model, input_tensor, output_path=model_info_path, device=str(device))

    stats["Parameters(M)"] = params / 1e6
    stats["FLOPs(G)"] = flops / 1e9

    average_time = inference_time / len(os.listdir(folder_path))
    stats["average_time(ms)"] = average_time
    if device == 'cpu':
        stats["device"] = str(device)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2  # covert to MiB
        device_info = f" CUDA: ({gpu_name}, {int(gpu_memory)}MiB)"
        stats["device"] = device_info

    with open(json_path, 'w') as json_file:
        json.dump(stats, json_file, indent=4)
    for k, v in stats.items():
        print(f'{k}: {v}')
