import numpy as np
import cv2 as cv
import os
from config import SUFFIX

# 获取文件名列表
def get_file_name(path):
    image_name_list = []
    for file in os.listdir(path):
        for item in SUFFIX:
            if file.endswith(item):
                image_name_list.append(os.path.join(path, file))
    return image_name_list


# 模板测试
path = 'E:\zhanglefu\dl_img\\new-bottle'
image_name_list = get_file_name(path)
for file in image_name_list:
    img = cv.imread(file)

    # img = cv.resize(img,(300,300))0000
    img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    h,s,v = cv.split(img_hsv)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurred = cv.blur(gray,(5,5))
    # cv.imshow('blurred',blurred)
    _,thresh_inv = cv.threshold(blurred,230,255,cv.THRESH_BINARY_INV)
    _,thresh = cv.threshold(blurred,230,255,cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(5,5))
    # closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    closed = cv.dilate(thresh,None,iterations=2)
    closed = cv.erode(closed,None,iterations=2)
    imgs,cnts,_ = cv.findContours(thresh_inv,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cs = sorted(cnts,key=cv.contourArea,reverse=True)
    cv.drawContours(img,[cs[0]],-1,(0,0,255),thickness=2)
    # print(thresh.dtype)
    cv.imshow('thresh',thresh)
    cv.imshow('thresh_inv',thresh_inv)
    # cv.imshow('closed',closed)
    cv.imshow('img',img)
    cv.waitKey(0)
# img = cv.imread('knife.jpg')
# cv.imshow('img',img)
# img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# h,s,v = cv.split(img_hsv)
# h_c = h.copy()
# for i in range(h.shape[0]):
#     for j in range(h.shape[1]):
#         if h[i][j]>124:
#             # print(h[i][j])
#             h_c[i][j] = 10
# new_img_hsv = cv.merge([h_c,s,v])
# img_bgr = cv.cvtColor(new_img_hsv,cv.COLOR_HSV2BGR)
# cv.imshow('img_bgr',img_bgr)
# cv.waitKey(0)