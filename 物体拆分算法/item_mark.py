'''
物体标记算法
'''
import cv2 as cv
import numpy as np
from tools import ItemExtractionTool

iet = ItemExtractionTool()

def item_mark(k,u,path):
    original = cv.imread(path)
    cp_original = original.copy()
    #存储需要标记的轮廓
    knife_mark_contour = []
    umbrella_mark_contour = []
    #存储用来标记物体的矩形
    knife_rect = []
    umbrella_rect = []
    #寻找刀的轮廓
    k = cv.cvtColor(k,cv.COLOR_BGR2GRAY)
    _,contours_k,h_k = cv.findContours(k,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for c in contours_k:
        area = cv.contourArea(c)
        if area>2500:
            knife_mark_contour.append(c)
            rect = cv.minAreaRect(c)
            box = np.int0(cv.boxPoints(rect))
            knife_rect.append(box)
    #寻找伞的轮廓
    u = cv.cvtColor(u,cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    u = cv.dilate(u,kernel,iterations=1)
    _,contours_u,h_u = cv.findContours(u,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,c in enumerate(contours_u):
        area = cv.contourArea(c)
        if area>5000:
            umbrella_mark_contour.append(c)
            rect = cv.minAreaRect(c)
            box = np.int0(cv.boxPoints(rect))
            umbrella_rect.append(box)
    #在原图中标记物体
    cv.drawContours(cp_original,knife_rect,-1,(0,0,255),2)
    cv.drawContours(cp_original,umbrella_rect,-1,(0,255,0),2)
    cv.imshow('item_mark',cp_original)
    #将原图中的刀和伞标记到新图
    k_bg = np.zeros((original.shape[0],original.shape[1]),dtype='uint8')
    cv.drawContours(k_bg, knife_mark_contour, -1, (255, 255, 255), cv.FILLED)
    knife = cv.bitwise_and(original,original,mask=k_bg)
    knife = iet.keep_blue_item(knife)

    u_bg = np.zeros((original.shape[0],original.shape[1]),dtype='uint8')
    cv.drawContours(u_bg, umbrella_mark_contour, -1, (255, 255, 255), cv.FILLED)
    umbrella = cv.bitwise_and(original,original,mask=u_bg)
    umbrella = iet.keep_blue_item(umbrella)
    return knife,umbrella
