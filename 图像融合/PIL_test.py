'''
背景透明处理
'''

# !/usr/bin/env python

# coding=utf-8


from PIL import Image, ImageDraw, ImageFont
import cv2 as cv
import numpy as np
def transparent_back(img):
    img = img.convert('RGBA')
    L,H = img.size
    color_0 = img.getpixel((0,0))
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1]+(0,)
                img.putpixel(dot,color_1)
    return img

if __name__ == '__main__':
    img = Image.open('bg_thresh.png')
    img = transparent_back(img)
    img = cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)
    print(img)
    cv.imwrite('bg_thresh2.png',img)
    # img.save('bg_thresh2.png')
