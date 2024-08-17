import cv2
import numpy as np
def ldr(src_data, alpha, U):
    """
    对比度增强函数，基于二维直方图的层次差分表示方法。

    参数：
        src_data : 输入数据，可以是2D直方图或灰度图像。
        alpha    : 控制增强程度的参数。
        U        : 方程（31）中的矩阵，如果提供，可以节省计算时间。

    返回值：
        out : 增强后的图像。
    """
    R, C = src_data.shape
    if R == 255 and C == 255:
        h2D_in = src_data
    else:
        in_Y = src_data
        # 无序2D直方图获取
        h2D_in = np.zeros((256, 256))
        for j in range(1, R + 1):
            for i in range(1, R + 1):
                ref = in_Y[j - 1, i - 1]
                if j != R:
                    trg = in_Y[j, i - 1]
                    h2D_in[np.maximum(trg, ref), np.minimum(trg, ref)] += 1
                if i != C:
                    trg = in_Y[j - 1, i]
                    h2D_in[np.maximum(trg, ref), np.minimum(trg, ref)] += 1
        del ref, trg

    # 层内优化
    D = np.zeros((255, 255))
    s = np.zeros((255, 1))

    # 迭代开始
    for layer in range(1, 255):
        h_l = np.zeros((256 - layer, 1))
        tmp_idx = 1
        for j in range(1 + layer, 257):
            i = j - layer
            h_l[tmp_idx - 1, 0] = np.log(h2D_in[j - 1, i - 1] + 1)
            tmp_idx += 1
        del tmp_idx

        s[layer - 1, 0] = np.sum(h_l)

        # 如果h_l中所有元素都为零，则跳过
        if s[layer - 1, 0] == 0:
            continue

        # 卷积
        m_l = np.convolve(np.squeeze(h_l), np.ones((layer,)))  # 方程（30）
        d_l = (m_l - np.amin(m_l)) / U[:, layer - 1]  # 方程（33）

        if np.sum(d_l) == 0:
            continue

        D[:, layer - 1] = d_l / sum(d_l)

    # 层间聚合
    W = (s / np.amax(s)) ** alpha  # 方程（23）
    d = np.matmul(D, W)  # 方程（24）

    # 重构转换函数
    d /= np.sum(d)  # 归一化
    tmp = np.zeros((256, 1))
    for k in range(1, 255):
        tmp[k] = tmp[k - 1] + d[k - 1]

    x = (255 * tmp).astype(np.uint8)
    out = x[src_data]
    return out

# 加载灰度图像
input_image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 设置对比度增强参数
alpha = 0.5
U = np.random.rand(255, 255)  # 在实际情况下，你需要提供合适的U矩阵

# 调用 ldr 函数进行对比度增强
enhanced_image = ldr(input_image, alpha, U)

# 显示原始图像和增强后的图像
cv2.imshow('Original Image', input_image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
