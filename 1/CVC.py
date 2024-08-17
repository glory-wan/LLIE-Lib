import cv2
import numpy as np
import matplotlib.pyplot as plt

def cvc(image, alpha=1.0, beta=0.0):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)

    # 计算对比度变化曲线
    hist, _ = np.histogram(gradient.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    contrast_curve = (cdf.max() - cdf) / cdf.max()

    # 根据对比度变化曲线调整图像对比度
    adjusted_gray = np.clip(alpha * gray + beta * contrast_curve[gray], 0, 255).astype(np.uint8)

    return adjusted_gray

# 读取图像
image = cv2.imread("input.jpg")

# 应用CVC增强
enhanced_image = cvc(image)

# 显示原始图像和增强后的图像
cv2.imshow("Enhanced image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
