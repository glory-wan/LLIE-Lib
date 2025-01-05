from LibLlie.deelLearning.utils import scriptDL

if __name__ == '__main__':
    img = scriptDL(
        model='Zero-DCE',
        model_path=r'D:\BaiduSyncdisk\code\LibLlie\LibLlie\models\SCI\difficult.pt',
        input=r'assets/input.jpg',
        # input=r'assets/DL_test',
        output_dir=r"C:\Users\13011\Desktop\结果图像",
        # save_format='jpg',

        # These two parameters only work if input is a file
        save_image=True,
        show_image=False,

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
