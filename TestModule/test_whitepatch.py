import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread, imshow
from matplotlib.patches import Rectangle

# 显示图片并标记出图中感兴趣的部分的代码，代码如下:
from skimage import io
import matplotlib.pyplot as plt

image = io.imread('Samples/RCB1763013_S1/RCB1763013_S1.bmp')



def calc_color_overcast(image):
    # Calculate color overcast for each channel
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Create a dataframe to store the results
    channel_stats = pd.DataFrame(columns=['Mean', 'Std', 'Min', 'Median', 'P_80', 'P_90', 'P_99', 'Max'])

    # Compute and store the statistics for each color channel
    for channel, name in zip([red_channel, green_channel, blue_channel], ['Red', 'Green', 'Blue']):
        mean = np.mean(channel)
        std = np.std(channel)
        minimum = np.min(channel)
        median = np.median(channel)
        p_80 = np.percentile(channel, 80)
        p_90 = np.percentile(channel, 90)
        p_99 = np.percentile(channel, 99)
        maximum = np.max(channel)

        channel_stats.loc[name] = [mean, std, minimum, median, p_80, p_90, p_99, maximum]
    print(channel_stats)
    return channel_stats
calc_color_overcast(image)