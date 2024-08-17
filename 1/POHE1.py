import cv2
import numpy as np


def integral_image(image):
    """
    计算积分图像。
    """
    integral_img = np.zeros_like(image, dtype=np.uint32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            integral_img[y, x] = image[y, x] + (integral_img[y, x - 1] if x > 0 else 0) \
                                 + (integral_img[y - 1, x] if y > 0 else 0) \
                                 - (integral_img[y - 1, x - 1] if (x > 0 and y > 0) else 0)
    return integral_img


def pohe(image, window_size=15):
    """
    使用POHE方法进行局部对比度增强。
    """
    # 计算积分图像
    integral_img = integral_image(image)

    # 计算窗口大小的一半
    half_window = window_size // 2

    # 创建输出图像
    output_image = np.zeros_like(image)

    # 遍历图像像素
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # 计算窗口的边界
            x1 = max(0, x - half_window)
            x2 = min(image.shape[1] - 1, x + half_window)
            y1 = max(0, y - half_window)
            y2 = min(image.shape[0] - 1, y + half_window)

            # 计算窗口内像素的累积和
            sum_in_window = integral_img[y2, x2] - integral_img[y2, x1 - 1] \
                            - integral_img[y1 - 1, x2] + integral_img[y1 - 1, x1 - 1]

            # 计算窗口内像素的数量
            num_pixels = (x2 - x1 + 1) * (y2 - y1 + 1)

            # 计算窗口内像素的平均灰度值
            mean_intensity = sum_in_window / num_pixels

            # 计算窗口内像素的标准差
            variance = np.mean((image[y1:y2 + 1, x1:x2 + 1] - mean_intensity) ** 2)

            # 计算局部对比度增强的结果
            output_image[y, x] = ((image[y, x] - mean_intensity) * np.sqrt(128 / variance) + mean_intensity).astype(
                np.uint8)

    return output_image


# 读取图像
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 对图像应用POHE
output_image = pohe(image)

# 显示原始图像和处理后的图像
cv2.imshow('Original Image', image)
cv2.imshow('POHE Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
