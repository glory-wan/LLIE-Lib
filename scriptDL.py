import LLIELib

if __name__ == '__main__':
    LLIELib.predict(
        './assets/input.jpg',
        model='Zero_DCE',
        weight='./weights/Zero_DCE/Zero_DCE.pth',

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
            Zero_DCE | SCI-easy | SCI-medium | SCI-difficult

    """
