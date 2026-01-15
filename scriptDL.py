import LLIELib

if __name__ == '__main__':
    LLIELib.predict(
        './assets/input.jpg',
        model='Zero-DCE',
        checkpoint='./checkpoints/Zero-DCE/Zero-DCE.pth',

        # following parameters are alternative
        # save_dir='results',
        # show_image=True,
        # save_format=None,
        # save_image=True,
        # gpu=0,
        # width=None,
        # height=None,
        # name_Suffix='',
    )

    """
        Quickly start:

        supported model:
            Zero-DCE | SCI-easy | SCI-medium | SCI-difficult

    """
