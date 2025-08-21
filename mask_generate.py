import cv2
import numpy as np

def black_background_to_binary(input_path, output_path, black_thresh=30, denoise=True):
    """
    将黑色背景转为纯黑(0)、其他内容转为纯白(1)的二值化处理
    
    参数：
        input_path: 输入图片路径
        output_path: 输出图片路径（推荐.png格式）
        black_thresh: 黑色阈值(0-255)，默认30
        denoise: 是否启用降噪，默认True
    """
    try:
        # 读取图片（使用IMREAD_UNCHANGED保留原始通道信息）
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("图片读取失败，请检查路径和文件格式")

        # 处理4通道图片（如PNG透明背景）
        if img.shape[2] == 4:
            alpha_channel = img[:,:,3]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # 将透明区域视为黑色背景
            black_mask = (alpha_channel < 50) | (np.mean(img, axis=2) < black_thresh)
        else:
            # 常规3通道图片处理
            black_mask = np.mean(img, axis=2) < black_thresh

        # 生成二值图（黑色背景=0，其他=1）
        binary = np.where(black_mask, 0, 1).astype(np.uint8) * 255

        # 降噪处理（可选）
        if denoise:
            binary = cv2.morphologyEx(
                binary, 
                cv2.MORPH_OPEN, 
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
            
            # 保留最大连通区域（移除小噪点）
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            if num_labels > 1:
                max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                binary = np.where(labels == max_label, 255, 0).astype(np.uint8)

        # 保存为1位深度的PNG（最小文件体积）
        cv2.imwrite(output_path, binary, [cv2.IMWRITE_PNG_BILEVEL, 1])
        
        print(f"处理成功：黑色背景阈值={black_thresh}，输出尺寸={binary.shape}")
        return True

    except Exception as e:
        print(f"处理失败：{str(e)}")
        return False

# 使用示例
input_img = "D:\\RobustPhotometricStereo\\data\\blender8\\render_output\\test_render.png"  # 支持jpg/png等格式
output_img = "D:\\RobustPhotometricStereo\\data\\blender8\\render_output\\mask.png" 

# 参数调优建议：
# 1. 背景不够黑 → 降低black_thresh
# 2. 内容被误认为背景 → 提高black_thresh
# 3. 有小噪点 → 启用denoise
success = black_background_to_binary(
    input_path=input_img,
    output_path=output_img,
    black_thresh=5,  # 典型值范围15-50
    denoise=True
)