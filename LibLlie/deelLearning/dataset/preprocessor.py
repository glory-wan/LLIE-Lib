import torch
from torchvision.transforms import v2


# from LibLlie.deelLearning.dataset.imageReader import ImageReader

class defaultTrans(v2.Compose):
    def __init__(self, size):
        super(defaultTrans, self).__init__([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size),
        ])


class Zero_DCE_Trans(v2.Compose):
    def __init__(self, device, size):
        super(Zero_DCE_Trans, self).__init__(
            [Zero_DCE(device, size)],
        )


class Zero_DCE(object):
    def __init__(self, device, size):
        self.device = device
        self.transform = v2.Compose([  # 转换为PyTorch张量并归一化
            v2.ToImage(),
            v2.Resize(size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])

    def __call__(self, img):
        img_lowlight = self.transform(img)  # 应用预处理
        img_lowlight = img_lowlight.to(self.device)  # 添加批次维度
        return img_lowlight

# if __name__ == '__main__':
#     imReader= ImageReader()
#     img = imReader(r'D:\LAB\LLIE-Lib\assets\input.jpg')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     transform = Zero_DCE_Trans(device)
#     output = transform(img)
#     print(output.shape)
