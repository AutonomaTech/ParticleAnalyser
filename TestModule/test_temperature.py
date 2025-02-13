import cv2
import numpy as np
import math
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


def adjust_temperature(image, from_temp, to_temp):
    # 计算原始色温和目标色温的RGB增益
    from_rgb = kelvin_to_rgb(from_temp)
    to_rgb = kelvin_to_rgb(to_temp)
    balance = to_rgb / from_rgb

    # 应用增益
    adjusted = (image * balance).clip(0, 255).astype(np.uint8)

    return adjusted
# 读取图片

image = cv2.imread('Samples/RCB1489190_S1/RCB1489190_S1.bmp')
result_image = adjust_temperature(image, 2272, 5753.9)
# 显示结果
cv2.imshow('Original', image)
cv2.imshow('Temperature Adjusted', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图片
cv2.imwrite('RCB1763013_S1.png', result_image)