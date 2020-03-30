'''
物体分割算法
主要使用区域增长算法,利用像素点的灰度差达到物体分割效果
聚类算法可以将同类物品区别于其他物品
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tools import ItemExtractionTool
#区域增长
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
    def getY(self):
        return self.y
#计算灰度差值
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y])-int(img[tmpPoint.x,tmpPoint.y]))
#选取区域增长方向
def selectConnects(p):
    if p!=0:
        connects = [Point(-1,-1),Point(0,-1),Point(1,-1),Point(1,0),
                    Point(1,1),Point(0,1),Point(-1,1),Point(-1,0)]
    else:
        connects = [Point(0,-1),Point(1,0),Point(0,1),Point(-1,0)]
    return connects
#区域增长核心算法
def regionGrow(img,seeds,thresh,p=1):
    height= img.shape[0]
    weight = img.shape[1]
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while len(seedList)>0:
        currentPoint = seedList.pop(0)
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x+connects[i].x
            tmpY = currentPoint.y+connects[i].y
            if tmpX<0 or tmpY<0 or tmpX>=height or tmpY>=weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff<thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark

# kmeans聚类
def km(img):
    img1 = img.reshape((img.shape[0]*img.shape[1],1))
    img1 = np.array(img1,np.float32)
    criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,50,1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv.kmeans(img1,3,None,criteria,10,flags)
    img2 = labels.reshape((img.shape[0],img.shape[1]))
    return img2

#形态学膨胀处理
def dilate(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    after_dilate = cv.dilate(image, kernel, iterations=1)
    return after_dilate
#腐蚀
def erode(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    after_erode = cv.erode(image, kernel, iterations=2)
    return after_erode


