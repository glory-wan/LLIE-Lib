import cv2
import numpy as np

def adaptive_dhe(image, tile_size=(8, 8)):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用预处理步骤，例如直方图均衡化
    gray = cv2.equalizeHist(gray)

    # 分割图像为小块
    h, w = gray.shape
    block_h = h // tile_size[0]
    block_w = w // tile_size[1]

    # 存储增强后的图像
    enhanced_image = np.zeros_like(gray)

    # 对每个小块应用直方图均衡化
    for i in range(tile_size[0]):
        for j in range(tile_size[1]):
            block = gray[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]

            # 计算每个小块的直方图
            hist, _ = np.histogram(block.flatten(), 256, [0, 256])

            # 计算累积直方图
            cdf = hist.cumsum()

            # 找到像素值范围的最小和最大值
            cdf_min = np.min(cdf)
            cdf_max = np.max(cdf)

            # 计算动态的clip_limit
            clip_limit = 2.0 * (cdf_max - cdf_min) / block.size

            # 使用CLAHE进行直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced_block = clahe.apply(block)

            # 将增强后的小块放回图像中
            enhanced_image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w] = enhanced_block

    return enhanced_image

# 读取图像
image = cv2.imread("input.jpg")

# 应用自适应DHE增强
enhanced_image = adaptive_dhe(image)

# 显示原始图像和增强后的图像
cv2.imshow("Enhanced image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
