from config import *
import cv2 as cv
import os
import numpy as np
import random


# 获取文件名列表
def get_image_name(path):
    image_name_list = []
    for file in os.listdir(path):
        for item in SUFFIX:
            if file.endswith(item):
                image_name_list.append(os.path.join(path, file))
    return image_name_list


# 图片预处理
def image_pretreatment(img, flag=False):
    # img = cv.imread(image_path)
    c_img = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    if flag == True:
        img = img[:, 2:width]
    # else:
    #     img = cv.resize(img, (100, 100))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gradX = cv.Sobel(gray, cv.CV_32F, dx=1, dy=0, ksize=-1)
    # gradY = cv.Sobel(gray, cv.CV_32F, dx=0, dy=1, ksize=-1)
    # gradient = cv.subtract(gradX, gradY)
    # gradient = cv.convertScaleAbs(gradient)
    # 去噪
    blurred = cv.blur(gray, (5, 5))
    # cv.imshow('blurred',blurred)
    _, thresh = cv.threshold(blurred, 90, 255, cv.THRESH_TRUNC)
    _, thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY_INV)

    # cv.imshow('thresh',thresh)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # cv.imshow("closed", closed)
    # #腐蚀与膨胀
    closed = cv.erode(thresh, kernel, iterations=1)
    closed = cv.dilate(closed, kernel1, iterations=2)
    # cv.imshow('closed',closed)
    # 绘制包所在位置的矩形边框
    # image1, cnts, _ = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(c_img,cnts,-1,(255,0,0),3)
    # cs = sorted(cnts, key=cv.contourArea, reverse=True)
    # for i,c in enumerate(cs):
    #     if i<3:
    #         x, y, w, h = cv.boundingRect(c)
    #         rect = cv.minAreaRect(c)
    #         box = cv.boxPoints(rect)
    #         box = np.int0(box)
    #         print(box)
    #         # cv.drawContours(c_img, c, 0, (255, 0, 0), 3)
    #         cv.drawContours(c_img,[box],0,(255,0,0),2)
    #         # cv.rectangle(c_img,(x,y),(x+w,y+h),(255,0,0),2)
    # cv.imshow('1111', c_img)
    # cv.waitKey(0)
    return img, closed


#处理刀的图片
def knife_pretreatment(img, flag=False):

    c_img = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    if flag == True:
        img = img[:, 2:width]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gradX = cv.Sobel(gray, cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv.Sobel(gray, cv.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv.subtract(gradX, gradY)
    gradient = cv.convertScaleAbs(gradient)
    # 去噪
    blurred = cv.blur(gradient, (5, 5))
    _, thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 25))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # cv.imshow("closed", closed)
    # #腐蚀与膨胀
    closed = cv.erode(closed, None, iterations=2)
    closed = cv.dilate(closed, None, iterations=2)
    # 绘制包所在位置的矩形边框
    # image1, cnts, _ = cv.findContours(closed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(c_img,cnts,-1,(255,0,0),3)
    # cs = sorted(cnts, key=cv.contourArea, reverse=True)
    # for i,c in enumerate(cs):
    #     if i<3:
    #         x, y, w, h = cv.boundingRect(c)
    #         cv.rectangle(c_img,(x,y),(x+w,y+h),(255,0,0),2)
    # cv.imshow('1111', c_img)
    # cv.waitKey(0)
    return img, closed




# 寻找轮廓
def find_contours(image, n, flag=True):
    image1, cnts, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cs = sorted(cnts, key=cv.contourArea, reverse=True)
    # print(len(cs))
    # 寻找裸刀轮廓
    if flag:
        x, y, w, h = cv.boundingRect(cs[0])
        return (x, y, x + w, y + h),cs
    # 寻找可放置物体的区域轮廓
    else:
        # 保存所有可放置物体的区域的列表
        l = []
        for i, c in enumerate(cs):
            if i < n:
                x, y, w, h = cv.boundingRect(c)
                # rect = cv.minAreaRect(c)
                # box = cv.boxPoints(rect)
                # box = np.int0(box)
                l.append((x,y,x+w,y+h))
        return l


def nothing(x):
    pass


# 改变图片的颜色
def change_image_color(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.namedWindow('winName')
    cv.createTrackbar('LowerbH', 'winName', 0, 255, nothing)
    cv.createTrackbar('LowerbS', 'winName', 0, 255, nothing)
    cv.createTrackbar('LowerbV', 'winName', 0, 255, nothing)
    cv.createTrackbar('UpperbH', 'winName', 255, 255, nothing)
    cv.createTrackbar('UpperbS', 'winName', 255, 255, nothing)
    cv.createTrackbar('UpperbV', 'winName', 255, 255, nothing)
    while True:
        # 函数cv.getTrackbarPos()范围当前滑块对应的值    
        lowerbH = cv.getTrackbarPos('LowerbH', 'winName')
        lowerbS = cv.getTrackbarPos('LowerbS', 'winName')
        lowerbV = cv.getTrackbarPos('LowerbV', 'winName')
        upperbH = cv.getTrackbarPos('UpperbH', 'winName')
        upperbS = cv.getTrackbarPos('UpperbS', 'winName')
        upperbV = cv.getTrackbarPos('UpperbV', 'winName')
        # 得到目标颜色的二值图像，用作cv.bitwise_and()的掩模
        lower = (lowerbH, lowerbS, lowerbV)
        upper = (upperbH, upperbS, upperbV)
        img_target = cv.inRange(img, lower, upper)
        # 输入图像与输入图像在掩模条件下按位与，得到掩模范围内的原图像
        img_specifiedColor = cv.bitwise_and(img, img, mask=img_target)
        # 同时转化为BGR,对比查看
        image = cv.cvtColor(img_specifiedColor, cv.COLOR_HSV2BGR)
        imgs = np.hstack([image, img_specifiedColor])
        cv.imshow('winName', imgs)
        if cv.waitKey(1) == ord('q'):
            return lower, upper


# 图片旋转
def image_rotation(path, angle, center=None, scale=1.0):
    img = cv.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    if center is None:
        center = (width // 2, height // 2)
    M = cv.getRotationMatrix2D(center, angle, scale)
    # cos = np.abs(M[0,0])
    # sin = np.abs(M[0,1])
    # nW = int((height*sin)+(width*cos))
    # nH = int((height*cos)+(width*sin))
    # M[0,2] += (nW/2)-cX
    # M[1,2] += (nH/2)-cY
    rotated = cv.warpAffine(img, M, (width, height))
    cv.imshow('r', rotated)
    cv.waitKey(0)
    return rotated

# 图片融合
# 找到放置物体的位置
def find_area(image,background,l,t=None):
    '''

    :param image:
    :param background:
    :param l: 背景图中可放置物体的区域的外接矩形左上角坐标和宽高
    :return:
    '''
    # image = cv.resize(image,(100,100),interpolation=cv.INTER_AREA)
    area_list = []
    i_h = image.shape[0]
    i_w = image.shape[1]
    # print(i_h,i_w)


    mask_list = []
    for item in l:
        mask = np.ones_like(background) * 255
        area_top_left_x = item[0]
        area_top_left_y = item[1]
        area_bottom_right_x = item[2]
        area_bottom_right_y = item[3]
        # 可放置物体的区域
        # area = background[area_top_left_y:area_bottom_right_y,area_top_left_x:area_bottom_right_x]
        # area_list.append(area)
        # 在指定范围内随机生成物体区域左上角坐标
        # startX = random.randint(area_top_left_x,area_bottom_right_x)
        # startY = random.randint(area_top_left_y,area_bottom_right_y)
        # print(startX,startY)
        # 在指定区域内放置物体

        # image = image[t[1]:t[3],t[0]:t[2]]

        # image = cv.resize(image, (area_bottom_right_x-area_top_left_x,
        #                           area_bottom_right_y-area_top_left_y))
        image = cv.resize(image,(0,0),fx=0.5,fy=0.5,interpolation=cv.INTER_NEAREST)

        # if area_bottom_right_x > area_bottom_right_y:
        #     r_h = area_bottom_right_y
        #     r_w = int((i_w / i_h) * r_h)
        #     image = cv.resize(image, (r_w, r_h))
        #     mask[area_top_left_y:area_top_left_y+r_h,area_bottom_right_x:area_bottom_right_x+r_w]=image
        #     mask_list.append(mask)
        #
        # elif area_bottom_right_x < area_bottom_right_y:
        #     r_w = area_bottom_right_x
        #     r_h = int((i_h / i_w) * r_w)
        #     image = cv.resize(image, (r_w, r_h))
        #     mask[area_top_left_y:area_top_left_y + r_h, area_bottom_right_x:area_bottom_right_x + r_w] = image
        #     mask_list.append(mask)

        # i_h = image.shape[0]
        # i_w = image.shape[1]
        # print(i_h,i_w)

        # mask[area_top_left_y:area_bottom_right_y,area_top_left_x:area_bottom_right_x] = image
        mask[area_top_left_y:area_top_left_y+image.shape[0],area_top_left_x+50:area_top_left_x+image.shape[1]+50] = image
        mask_list.append(mask)
    return mask_list

# 放物体
# def put_item(image,t,l):
#     image = image[t[1]:t[3],t[0]:t[2]]
#     height = image.shape[0]
#     width = image.shape[1]
#     mask[height,width] = image



# 图片融合
# def image_fusion(background,cnts,mask):
#     cv.drawContours(background,cnts,-1,(0,0,0),cv.FILLED)
    # cv.imshow('bg',background)
    # cv.waitKey(0)
    # res = cv.add(mask,background)
    # res = cv.add(background,mask)
    # return res

# 调节白色背景图片的对比度,将背景趋于透明
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    src2 = np.ones([h, w, ch], src1.dtype) * 255
    dst = cv.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    # cv2.imshow("con-bri-demo", dst)
    return dst

if __name__ == '__main__':
    l = get_image_name(r'E:\vid')
    for item in l:
        img = cv.imread(item)
        image_pretreatment(img,flag=True)
    pass

