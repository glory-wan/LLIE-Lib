import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from LibLlie.deelLearning.config import models
from LibLlie.deelLearning.dataset.basedataset import baseDataSet
from LibLlie.deelLearning.utils.utils import log_info_env, save_image


def scriptDL(
        model=None,
        model_path=None,
        input_dir=None,
        output_dir=None,
        save_format='png',
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Now, you are using '{device}' to run model!")

    model = models[model](model_path).to(device)
    transform = None
    testDataset = baseDataSet(input_dir=input_dir, transform=transform)
    testImages = DataLoader(testDataset, batch_size=1, pin_memory=True, num_workers=os.cpu_count())

    model.eval()
    with torch.no_grad():
        for imTensor, name in tqdm(testImages):
            imTensor = imTensor.to(device)
            i, output = model(imTensor)
            name = name[0] + '.' + save_format
            save_path = os.path.join(output_dir, name)
            save_image(output, save_path)


if __name__ == '__main__':
    scriptDL(
        model='SCI-easy',
        model_path='../../models/SCI/easy.pt',
        input_dir=r'D:\Code\pycode\dataset_all\DSL\demo',
        output_dir=r"D:\Code\pycode\dataset_all\DSL\test",
    )
