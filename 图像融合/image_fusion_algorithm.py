import cv2 as cv
import numpy as np
from PIL import Image

# 线性方法将刀具图片的背景变为白色
def updateAlphaBeta(img,alpha,beta):
    img = np.uint8(np.clip((alpha*img+beta),0,255))
    return img

# 将背景变为透明
def transparent_back(img):
    img = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB))
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
    img = cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)
    return img

def knife_pretreatment(img):
    c_img = img.copy()
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
    # cv.imshow('closed',closed)
    image1, cnts, _ = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(cnts))
    cs = sorted(cnts, key=cv.contourArea, reverse=True)
    cv.drawContours(c_img, cs, -1, (255, 0, 0), 3)
    # cv.imshow('c_img',c_img)
    # cv.waitKey(0)
    return img,closed

def image_fusion(knife,background):
    # 选择一块区域放置物体
    roi = background[100:300,200:400]
    # 灰度化
    knife_gray = cv.cvtColor(knife,cv.COLOR_BGR2GRAY)
    # 二值化
    ret,mask = cv.threshold(knife_gray,175,255,cv.THRESH_BINARY)
    # 取反
    mask_inv = cv.bitwise_not(mask)
    cv.imshow('mask_inv',mask_inv)
    # 在背景上画出值为0的刀具区域
    background1 = cv.bitwise_and(roi,roi,mask=mask)
    cv.imshow('background1',background1)
    # 取roi和mask_inv中不为零的值对应的像素值
    knife1 = cv.bitwise_and(knife,knife,mask=mask_inv)
    cv.imshow('knife1',knife1)

    dst = cv.add(background1,knife1)
    background[100:300,200:400] = dst
    cv.imshow('res',background)
    cv.waitKey(0)

# 图像旋转和镜像
def rotation_and_flip(img,angle,flip_style,scale=1.0,flag=False):
    # flip_style 1 水平镜像 0 垂直镜像 -1 对角镜像
    if flag:
        img = cv.flip(img,flip_style,dst=None)
    height = img.shape[0]
    width = img.shape[1]
    (cX, cY) = (width // 2, height // 2)
    M = cv.getRotationMatrix2D((cX, cY), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv.warpAffine(img, M, (nW, nH))

def nothing(x):
    pass

def change_image_color(image):
    img = cv.resize(image,(300,300))
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img)
    h = cv.cvtColor(h,cv.COLOR_GRAY2BGR)
    s = cv.cvtColor(s,cv.COLOR_GRAY2BGR)
    v = cv.cvtColor(v,cv.COLOR_GRAY2BGR)
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
        img_target = cv.cvtColor(img_target,cv.COLOR_GRAY2BGR)
        image = cv.cvtColor(img_specifiedColor, cv.COLOR_HSV2BGR)
        imgs = np.hstack([image, img_specifiedColor,img_target])
        imgs1 = np.hstack([h,s,v])
        imgs2 = np.vstack([imgs,imgs1])
        cv.imshow('winName', imgs2)
        if cv.waitKey(1) == ord('q'):
            return lower, upper

if __name__ == '__main__':
    knife = cv.imread('knife.jpg')
    # knife = cv.resize(knife,(200,200))
    # bg = cv.imread('bg.jpg')
    # knife = updateAlphaBeta(knife,1.5,0)
    # image_fusion(knife,bg)
    l,u = change_image_color(knife)

