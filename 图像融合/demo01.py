import cv2 as cv
import numpy as np

# 刀具图像预处理并制作其二值化图像
def knife_pretreatment(img, threshold):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 去噪
    blurred = cv.blur(gray, (5, 5))
    # 二值化制作模板
    _, thresh = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY_INV)
    image1, cnts, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return img,thresh,0,0,0,0
    cs = sorted(cnts, key=cv.contourArea, reverse=True)
    x, y, w, h = cv.boundingRect(cs[0])
    # 修改原图,只保留刀具部分
    # cv.imshow('knife', img)
    # cv.imshow('knife_thresh', thresh)
    img = img[y:y + h, x:x + w]
    # 修改模板,保留和图片大小一致的部分
    thresh = cv.cvtColor(thresh,cv.COLOR_GRAY2BGR)
    thresh = thresh[y:y + h, x:x + w]
    thresh = cv.cvtColor(thresh,cv.COLOR_BGR2GRAY)
    # 截取部分的长宽比
    ratio = h/w
    return img,thresh,x,y,w,h,ratio

# 制作反向模板
def knife_pretreatment_inv(img,threshold):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 去噪
    blurred = cv.blur(gray, (5, 5))
    # 二值化制作模板
    _, thresh = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY)
    # cv.imshow('inv',thresh)
    return thresh

# 背景预处理
def background_pretreatment(img,threshold,ksize=(3,3)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.blur(gray, (5, 5))
    _, thresh = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=ksize)
    closed = cv.erode(thresh, kernel, iterations=1)
    closed = cv.dilate(closed, kernel, iterations=1)
    return closed

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


