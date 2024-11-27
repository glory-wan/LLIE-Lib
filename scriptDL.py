from LibLlie.deelLearning.utils import scriptDL

if __name__ == '__main__':
    img = scriptDL(
        model='Zero-DCE',
        model_path='LibLlie/models/Zero-DCE/Zero-DCE.pth',
        input=r'assets/input.jpg',
        # input=r'assets/DL_test',
        output_dir=r'results',
        # save_format='jpg',

        # These two parameters only work if input is a file
        save_image=False,
        show_image=True,

        # following parameters are alternative
        # save_image=False,
        # show_image=True,
        # gpu=0,
        # batch_size=1,
        # output_height=512,
    )

    """
        Quickly start:

        supported model:
            Zero-DCE | SCI-easy | SCI-medium | SCI-difficult

    """
