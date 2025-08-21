import cv2
import numpy as np

# 读取图像
img = cv2.imread("D:\\RobustPhotometricStereo\\data\\blender6\\render_output\\image0000.png")

# 分离 RGB 通道
b, g, r = cv2.split(img)

# 计算绿色部分（绿色通道较强，红蓝通道较弱）
green_only = cv2.subtract(g, cv2.max(b, r))

# 二值化（阈值可根据实际情况调整）
_, binary = cv2.threshold(green_only, 50, 255, cv2.THRESH_BINARY)

# 保存结果
cv2.imwrite("D:\\RobustPhotometricStereo\\data\\blender6\\render_output\\mask.png", binary)