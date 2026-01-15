import cv2

def resize_keep_ratio(img, max_side=1024):
    """将图像缩放到最长边不超过max_side，保持比例"""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img.copy()
    scale = max_side / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

# 使用
img = cv2.imread("assets/input.jpg")
resized = resize_keep_ratio(img, 1024)
cv2.imwrite("assets/input2.jpg", resized)