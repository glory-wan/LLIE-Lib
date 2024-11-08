from LibLlie.deelLearning.utils.utils import scriptDL

if __name__ == '__main__':
    img = scriptDL(
        model='Zero-DCE',
        model_path=r'LibLlie/models/Zero-DCE/Zero-DCE.pth',
        input_dir=r'assets/DL_test/input',
        output_dir=r'assets/DL_test/output',
        save_format='jpg',

        batch_size=1,
        output_height=256,
    )

    """
        Quickly start:

        supported model:
            Zero-DCE | SCI-easy | SCI-medium | SCI-difficult

    """
