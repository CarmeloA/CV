'''
图片预处理
'''
import cv2 as cv
import numpy as np
# 图片增强
image = cv.imread('original.png')

# dst = cv.fastNlMeansDenoisingColored(image,None,10,10,7,21)
# Gaussian = cv.GaussianBlur(image,(5,5),0)
# sobelx = cv.Sobel(dst,cv.CV_64F,1,0,ksize=5)
# sobelx = cv.Sobel(sobelx,cv.CV_64F,1,0,ksize=5)
# sobely = cv.Sobel(dst,cv.CV_64F,0,1,ksize=5)
# sobely = cv.Sobel(sobely,cv.CV_64F,0,1,ksize=5)
# sobel = sobelx-sobely
# canny = cv.Canny(dst,50,240)
# # cv.imshow('sobelx',sobelx)
# # cv.imshow('sobely',sobely)
# cv.imshow('original',image)
# cv.imshow('dst',dst)
# cv.imshow('Guassian',Gaussian)
# cv.imshow('sobel',sobel)
# cv.imshow('canny',canny)
# cv.waitKey(0)
red = np.zeros_like(image)
green = np.zeros_like(image)
blue = np.zeros_like(image)
red[:,:,2] = image[:,:,2]
green[:,:,1] = image[:,:,1]
blue[:,:,0] = image[:,:,0]
hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

lower_blue = np.array([70, 43, 46])
upper_blue = np.array([124, 255, 255])
lower_orange = np.array([11, 43, 46])
upper_orange = np.array([25, 255, 255])
lower_green = np.array([35, 43, 46])
upper_green = np.array([77, 255, 255])

mask = cv.inRange(hsv,lower_blue,upper_blue)
res = cv.bitwise_and(image, image, mask=mask)


cv.imshow('win',hsv)
cv.imshow('win1',res)
cv.imshow('win2',mask)
cv.waitKey(0)