import numpy as np
import cv2

data = np.load('D:\\RobustPhotometricStereo\\data\\blender8\\render_output\\images\\image010.npy')

# 法线贴图的xyz分量通常在 [-1, 1] 范围，需要映射到 [0, 1] 再转到 [0, 255]
if data.dtype != np.uint8:
    # 将法线值从 [-1, 1] 转换到 [0, 1]
    data_normalized = (data + 1.0) / 2.0
    # 然后再从 [0, 1] 转换到 [0, 255]
    data_normalized = (data_normalized * 255).astype(np.uint8)

cv2.imwrite('converted_image.png', data_normalized)
print("彩色图已保存为 converted_image.png")