import torch
import os
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader

from LLIELib.deepLearning.config import models, parameters_dl
from LLIELib.deepLearning.utils import results_path, log_info_env, saveImg, showImg
from LLIELib.deepLearning.dataset.basedataset import PredictDataSet, single_data_loader
from LLIELib.deepLearning.dataset.preprocessor import predict_Trans


def predict(
        images,
        model=None,
        weight=None,
        save_dir=results_path,
        save_format=None,
        save_image=True,
        show_image=False,
        gpu=0,
        width=None,
        height=None,
        name_Suffix='',
):
    # env info
    log_info_env()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # load the model from weight
    try:
        print(f"\n Now, you are using '{device}' to run {model} model!\n")
        model = models[model]().to(device)
        model.load_state_dict(torch.load(weight, weights_only=True, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weight file '{weight}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred while loading model: {e}")
        sys.exit(1)
    
    # preprocess
    transform = predict_Trans()

    if os.path.isdir(images):
        preDataset = PredictDataSet(input_dir=images, transform=transform, size=(width, height))
        testImages = DataLoader(preDataset, batch_size=1, pin_memory=True, num_workers=(int(os.cpu_count()/2)))

        model.eval()
        with torch.no_grad():
            pbar = tqdm(testImages, desc=f'using {os.path.basename(weight)} to infer', unit='image')
            for imTensor, name in pbar:
                imTensor = imTensor.to(device)
                enResult = model(imTensor)
                enImg = enResult['enImg']

                if save_format is None:
                    save_format = name[1][0]

                saveImg(enImg, save_dir, name=f'{name[0][0]}{name_Suffix}', save_format=save_format)
                pbar.set_postfix({"image": f'{name[0][0]}{name_Suffix}{save_format}'})
                
    elif os.path.isfile(images):
        imTensor, name = single_data_loader(images, transform)
        model.eval()
        with torch.no_grad():
            imTensor = imTensor.to(device)
            enResult = model(imTensor)
            enImg = enResult['enImg']

        if save_image:
            if save_format is None:
                save_format = name[1]

            saveImg(enImg, save_dir, name=f'{name[0]}{name_Suffix}', save_format=save_format)
            print(f'have saved enhanced {images} to {save_dir}/{name[0]}{save_format}')
        if show_image:
            showImg(enImg, name=f'{name[0]}{name_Suffix}{save_format}')
    else:
        raise FileNotFoundError(f"\nInput path '{images}' is neither a valid file nor a directory. "
                                f"Please provide either:\n"
                                f"1. A directory containing images (current is_dir check: {os.path.isdir(images)})\n"
                                f"2. A single image file (current is_file check: {os.path.isfile(images)})\n"
                                f"3. Check if the path exists: {os.path.exists(images)}")


def command_DL():
    pta = parameters_dl()
    predict(
        images=pta.input,
        model=pta.model,
        weight=pta.weight,
        save_dir=pta.save_dir,
        save_format=pta.save_format,
        save_image=pta.save_image,
        show_image=pta.show_image,
        gpu=pta.gpu,
        name_Suffix=pta.name_Suffix,
    )