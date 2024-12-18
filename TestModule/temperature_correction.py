import numpy as np
from PIL import Image
import math
import time
import cv2

def convert_K_to_RGB(colour_temperature):
    """
    Converts from K to RGB, algorithm courtesy of
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    # range check
    if colour_temperature < 1000:
        colour_temperature = 1000
    elif colour_temperature > 40000:
        colour_temperature = 40000

    tmp_internal = colour_temperature / 100.0

    # red
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red

    # green
    if tmp_internal <= 66:
        tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green

    # blue
    if tmp_internal >= 66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue
    return (red , green , blue )

import matplotlib.pyplot as plt
def get_rgb_from_temperature(temp):
    """
    根据色温 (Kelvin) 计算白点的 RGB 值
    """
    temp = max(1000, min(temp, 40000)) / 100.0

    # 计算R分量
    if temp <= 66:
        r = 255
    else:
        tmpCalc = temp - 55
        r = 351.976905668057 + 0.114206453784165 * tmpCalc - 40.2536630933213 * np.log(tmpCalc)
        r = min(255, max(0, r))

    # 计算G分量
    if temp <= 66:
        tmpCalc = temp - 2
        g = -155.254855627092 -0.445969504695791 * tmpCalc + 104.492161993939 * np.log(tmpCalc)
        g = min(255, max(0, g))
    else:
        tmpCalc = temp - 50
        g = 325.449412571197 + 0.0794345653666234 * tmpCalc - 28.0852963507957 * np.log(tmpCalc)
        g = min(255, max(0, g))

    # 计算B分量
    if temp >= 66:
        b = 255
    else:
        if temp <= 19:
            b = 0
        else:
            tmpCalc = temp - 10
            b = -254.769351841209 + 0.827409606400739 * tmpCalc + 115.679944010661 * np.log(tmpCalc)
            b = min(255, max(0, b))

    return np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)

def get_rgb_from_temperature1(temp):
    temp = max(1000, min(temp, 40000)) / 100.0
    r, g, b = 0, 0, 0

    # Red
    if temp <= 66:
        r = 255
    else:
        tmpCalc = temp - 55
        r = 351.976905668057 + 0.114206453784165 * tmpCalc + -40.2536630933213 * np.log(tmpCalc)
        r = min(255, max(0, r))

    # Green
    if temp <= 66:
        tmpCalc = temp - 2
        g = -155.254855627092 + -0.445969504695791 * tmpCalc + 104.492161993939 * np.log(tmpCalc)
        g = min(255, max(0, g))
    else:
        tmpCalc = temp - 50
        g = 325.449412571197 + 7.94345653666234E-02 * tmpCalc + -28.0852963507957 * np.log(tmpCalc)
        g = min(255, max(0, g))

    # Blue
    if temp >= 66:
        b = 255
    else:
        if temp <= 19:
            b = 0
        else:
            tmpCalc = temp - 10
            b = -254.769351841209 + 0.827409606400739 * tmpCalc + 115.679944010661 * np.log(tmpCalc)
            b = min(255, max(0, b))
    return (r , g , b )



def color_error(rgb1, rgb2):
    diff = np.array(rgb1) - np.array(rgb2)
    return math.sqrt(np.sum(diff ** 2))
def kelvin_to_rgb(temp_kelvin):

    if temp_kelvin < 1000:
        temp_kelvin = 1000
    elif temp_kelvin > 40000:
        temp_kelvin = 40000

    tmp_internal =  temp_kelvin / 100.0

    # red
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red

    # green
    if tmp_internal <= 66:
        tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green

    # blue
    if tmp_internal >= 66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue

    # 创建RGB增益矩阵
    return np.clip([red, green, blue], 0, 255)

def get_temperature_from_rgb(target_r, target_g, target_b):
    target_rgb = (target_r, target_g, target_b)
    start_time = time.time()
    min_error = float('inf')
    best_temp = 1000

    # 全范围1K逐步搜索
    for temp in range(1000, 40001, 1):
        rgb = get_rgb_from_temperature(temp)
        err = color_error(rgb, target_rgb)
        if err < min_error:
            min_error = err
            best_temp = temp

    end_time = time.time()
    print("Searching consumption time: {:.2f}s".format(end_time - start_time))
    return best_temp, min_error


def estimate_temperature_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    arr = np.array(image)
    avg_rgb = np.mean(arr.reshape(-1, 3), axis=0)
    avg_r, avg_g, avg_b = avg_rgb

    # estimated_temp, error = get_temperature_from_rgb(avg_r, avg_g, avg_b)
    estimated_temp, error = get_temperature_from_rgb(avg_r, avg_g, avg_b)
    return estimated_temp, error


image_temp, err = estimate_temperature_from_image(r"C:\Users\LiCui\Desktop\ColorCorrectedImages\RCB1763362\RCB1763362_V1\RCB1763362_V1.bmp")
print("估计的原图图片色温:", image_temp, "K, 误差:", err)

def apply_temperature_to_image(image_path, target_temp, output_path):
    """
    将目标色温对应的RGB值作用于原始图片上，生成新的调整后图片。
    :param image_path: 原始图片路径
    :param target_temp: 目标色温(如 5753.9K)
    :param output_path: 调整后图片输出路径
    """
    # 打开原始图片并转为RGB
    image = Image.open(image_path).convert('RGB')
    arr = np.array(image, dtype=np.float32)

    # 计算原始图片的平均 RGB
    avg_original = np.mean(arr.reshape(-1, 3), axis=0)  # 原来的平均RGB

    # 取得目标色温对应的RGB值
    # target_rgb = convert_K_to_RGB(target_temp).astype(np.float32).reshape(3)
    target_rgb = convert_K_to_RGB(target_temp)
    # 如果原图平均值中有0，避免除零错误（极端情况）
    avg_original = np.maximum(avg_original, 1e-6)

    # 计算比例因子 scale = target_rgb / avg_original
    scale = target_rgb / avg_original

    # 将比例因子应用到整张图片
    # arr是[h, w, 3], 对每个通道分别相乘
    arr[..., 0] *= scale[0]
    arr[..., 1] *= scale[1]
    arr[..., 2] *= scale[2]

    # 限制像素值在0-255
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 生成新的图像并保存
    new_image = Image.fromarray(arr, mode='RGB')
    new_image.save(output_path)

    return output_path
# apply_temperature_to_image(r"C:\Users\LiCui\Desktop\Samples\RCB1763013\RCB1763013_S1\RCB1763013_S1.bmp",2500,'Samples/RCB1763013_S1/RCB1763013_S1.png')


def adjust_temperature(image, from_temp, to_temp):
    # 计算原始色温和目标色温的RGB增益
    from_rgb = get_rgb_from_temperature(from_temp)
    to_rgb = get_rgb_from_temperature(to_temp)
    balance = to_rgb / from_rgb

    # 应用增益
    adjusted = (image * balance).clip(0, 255).astype(np.uint8)

    return adjusted


temp = 2422.7 # 色温 (Kelvin)
rgb = get_rgb_from_temperature(temp)
r = rgb[..., 0]
g = rgb[..., 1]
b = rgb[..., 2]
print(f"色温 {temp}K 的 RGB 值是: R={r}, G={g}, B={b}")
temp = 5753.9# 色温 (Kelvin)
rgb = get_rgb_from_temperature(temp)
r = rgb[..., 0]
g = rgb[..., 1]
b = rgb[..., 2]
print(f"色温 {temp}K 的 RGB 值是: R={r}, G={g}, B={b}")
def blend_color(img_ori, white_point, alpha=0.8):
    '''
    Color fusion.
    Out = alpha * img_ori + (1 - alpha) * white_point
    '''
    img_dst = alpha * img_ori + (1 - alpha) * white_point
    img_dst[img_dst > 255] = 255
    img_dst = np.uint8(img_dst)
    return img_dst
# 注意，这里我们需要确保 img_ori 和 white_point 都是 NumPy 数组
img_ori = np.array([[255], [239], [231]])  # 注意要求RGB顺序
white_point = get_rgb_from_temperature(8000)  # 假设这个函数返回的是一个列表
white_point = np.array(white_point)  # 转换为 NumPy 数组



# # 或采用色适应假设：以的白点归一化. 这种方案不会显得图片偏白
img_dst = img_ori / white_point * 255
img_dst[img_dst > 255] = 255
img_dst = np.uint8(img_dst)


# image = cv2.imread(r"C:\Users\LiCui\Desktop\Samples\RCB1489190\RCB1489190_V19\RCB1489190_V19.bmp")
# result_image = adjust_temperature(image,  2648
# , 3000)
# # 显示结果
# cv2.imshow('Original', image)
# cv2.imshow('Temperature Adjusted', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 保存结果图片
# cv2.imwrite('Samples/RCB1489190_S1/RCB1489190_S1.png', result_image)