'''
图像融合算法工具模块
'''
from config import *
import cv2 as cv
import os
import numpy as np
import random

# 获取文件名列表
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1][1:] in SUFFIX:
                L.append(os.path.join(root, file))
    # print(L)
    return L

# 处理背景图片,并选定可以放置物体的区域
def get_background_roi(img,threshold=120):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 去噪
    blurred = cv.blur(gray, (5, 5))
    _, thresh = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY_INV)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.erode(thresh, kernel, iterations=1)
    closed = cv.dilate(closed, kernel1, iterations=2)
    cv.imshow('closed',closed)
    image1, cnts, _ = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return (0,0,0,0)
    cs = sorted(cnts, key=cv.contourArea, reverse=True)
    # 保存所有可放置物体的区域的列表
    x, y, w, h = cv.boundingRect(cs[0])
    # TODO 以后需要改成最小外切矩形
    # rect = cv.minAreaRect(c)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    return (x,y,w,h)

# 图片旋转
def image_rotation(img, angle, center=None, scale=1.0):
    height = img.shape[0]
    width = img.shape[1]
    (cX,cY) = (width//2,height//2)

    M = cv.getRotationMatrix2D((cX,cY), angle, scale)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int((height*sin)+(width*cos))
    nH = int((height*cos)+(width*sin))
    M[0,2] += (nW/2)-cX
    M[1,2] += (nH/2)-cY
    rotated = cv.warpAffine(img, M, (nW, nH))
    # cv.imshow('r', rotated)
    # cv.waitKey(0)
    return rotated

# 寻找背景图片中放置刀具的位置
def get_background_real_roi(t,knife,background):
    background_copy = background.copy()
    startX = None
    startY = None
    # 刀具图大小
    k_w = knife.shape[1]
    k_h = knife.shape[0]

    # 可放置刀具的范围的位置信息
    roi_top_left_x = t[0]
    roi_top_left_y = t[1]
    roi_w = t[2]-10
    roi_h = t[3]-10
    # 判断
    if k_h>max(roi_h,roi_w) and k_w>max(roi_h,roi_w):
        return background,0,0,0
    angle = 0
    # 旋转
    if ((k_h > roi_h) and (k_h < roi_w)) or ((k_w > roi_w) and (k_w < roi_h)):
        knife = image_rotation(knife, 90)
        k_w = knife.shape[0]
        k_h = knife.shape[1]
        angle = 90
    # 在roi中取出与刀具图大小一致的区域
    if (roi_w - k_w) >3 and (roi_h - k_h) > 3:
        startX = roi_top_left_x + random.randint(1,roi_w - k_w)
        startY = roi_top_left_y + random.randint(1,roi_h - k_h)
        background_real_roi = background_copy[startY:startY + k_h, startX:startX + k_w]
    # 如果物体图比较大就将其缩小
    # elif ((k_w - roi_w > 3) and (k_w//2 < roi_w)) and ((k_h - roi_h > 3)and(k_h//3 < roi_h)):
    #     knife = cv.resize(knife,(k_h//2,k_w//3))
    #     k_w = knife.shape[1]
    #     k_h = knife.shape[0]
    #     startX = roi_top_left_x + random.randint(1, k_w//2)
    #     startY = roi_top_left_y + random.randint(1, k_h//2)
    #     # startX = roi_top_left_x
    #     # startY = roi_top_left_y
    #     background_real_roi = background_copy[startY:startY + k_h, startX:startX + k_w]
    else:
        return background,0,0,0
    return background_real_roi, startX, startY, angle

# labels数据处理
def convert(startX,startY,w,h,img_w,img_h,flag=True):
    if flag:
        # 归一化后中心点坐标
        # mid_x = ((startX+startX+w)/2)/img_w
        # mid_y = ((startY+startY+h)/2)/img_h
        # 归一化后左上角坐标
        left_x = startX/img_w
        left_y = startY/img_h
        # 归一化后目标框
        bb_w = w/img_w
        bb_h = h/img_h
        # line = str(mid_x)+' '+str(mid_y)+' '+str(bb_w)+' '+str(bb_h)
        line = str(left_x)+' '+str(left_y)+' '+str(bb_w)+' '+str(bb_h)
    else:
        # xml文件需要的数据
        # 左上角坐标
        x_min = startX
        y_min = startY
        # 右下角坐标
        x_max = x_min+w
        y_max = y_min+h
        line = str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)
    return line








