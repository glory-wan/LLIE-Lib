import cv2
import numpy as np

def wthe(image, threshold=128, weight=0.5):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的直方图
    hist, _ = np.histogram(gray.flatten(), 256, [0, 256])

    # 计算累积直方图
    cdf = hist.cumsum()

    # 找到像素值范围的最小和最大值
    cdf_min = np.min(cdf)
    cdf_max = np.max(cdf)

    # 计算动态的clip_limit
    clip_limit = weight * (cdf_max - cdf_min)

    # 应用WTHE公式计算新的像素值
    new_pixels = np.where(gray <= threshold, (cdf[gray] - cdf_min) / clip_limit * threshold, (cdf_max - cdf[gray]) / clip_limit * (255 - threshold) + threshold)

    # 将像素值限制在0和255之间
    new_pixels = np.clip(new_pixels, 0, 255)

    # 将像素值转换为8位整数
    new_pixels = new_pixels.astype(np.uint8)

    return new_pixels

# 读取图像
image = cv2.imread("input.jpg")

# 应用WTHE增强
enhanced_image = wthe(image)

# 显示原始图像和增强后的图像
cv2.imshow("Enhanced image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
