import numpy as np
import cv2 as cv

knife_rgb = cv.imread('knife.jpg')
knife_rgb = cv.resize(knife_rgb, (300, 300))
knife_hsv = cv.cvtColor(knife_rgb, cv.COLOR_BGR2HSV)
# cv.imshow('knife_hsv',knife_hsv)
k_h, k_s, k_v = cv.split(knife_hsv)
# cv.imshow('k_h', k_h)
background_rgb = cv.imread('bg.jpg')
background_rgb = background_rgb[0:300, 100:400]
# cv.imshow('background',background)
background_hsv = cv.cvtColor(background_rgb, cv.COLOR_BGR2HSV)
b_h, b_s, b_v = cv.split(background_hsv)

k_mask = np.ones_like(k_h)
b_mask = np.ones_like(b_h)

# 两个列表存储模板需要的系数和坐标
k_mask_list = []
b_mask_list = []

for (k_height_i, k_height), (b_height_i, b_height) in zip(enumerate(k_h), enumerate(b_h)):
    for (k_width_i,k_width), (b_width_i,b_width) in zip(enumerate(k_height), enumerate(b_height)):
        k_width = int(k_width)
        b_width = int(b_width)
        if int(k_width + b_width) != 0:
            # 刀具图的系数
            k_c = k_width / int(k_width + b_width)
            # 背景图的系数
            b_c = b_width / int(k_width + b_width)
            # 重新赋值??
            # k_width = k_width * k_c
            # b_width = b_width * b_c
            # k_height[k_height_i] = k_width
            # b_height[b_height_i] = b_width
            # 存储模板用数据
            if k_c!=0:
                k_t = (k_height_i, k_width_i, k_c)
                k_mask_list.append(k_t)
            if b_c!=0:
                b_t = (b_height_i, b_width_i, b_c)
                b_mask_list.append(b_t)
print(k_mask_list)
print(len(k_mask_list))



