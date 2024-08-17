import cv2
import numpy as np


def bpdhe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的直方图
    hist, _ = np.histogram(gray.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_min = np.min(cdf)
    cdf_max = np.max(cdf)

    new_pixels = ((cdf[gray] - cdf_min) / (cdf_max - cdf_min)) * 255
    new_pixels = np.clip(new_pixels, 0, 255)
    new_pixels = new_pixels.astype(np.uint8)

    enhanced_image = np.zeros_like(gray)
    enhanced_image[:, :] = new_pixels.reshape(gray.shape)

    return enhanced_image


# 读取图像
image = cv2.imread("input.jpg")

# 应用BPDHE增强
enhanced_image = bpdhe(image)

# 显示原始图像和增强后的图像
cv2.imshow("Original", image)
cv2.imshow("Enhanced", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
