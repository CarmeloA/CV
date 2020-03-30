import cv2 as cv

import numpy as np


img = cv.imread('knife.png')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

ret, binary =cv.threshold(gray,127,255,cv.THRESH_BINARY)

binary,contours, hierarchy =cv.findContours(binary,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(len(contours))
C=[]
for c in contours:
    area = cv.contourArea(c)
    if area>800:
        C.append(c)
print(len(C))
cv.drawContours(img,C,-1,(255,255,255),2)
cv.imshow('img',img)
#cv2.drawContours(img,contours,-1,(0,0,255),3)

#cv2.imshow("img", img)

#cv2.waitKey(0)

#cv2.RETR_TREE

#print (type(contours))

#print (type(contours[0]))

#print (len(contours[0]))

#print (contours[0]-contours[0])

columns = []

for i in range(81):
    columns.append(C[0][i]-C[0][i - 1])

#print (len(columns))

#print (columns[1][0][0])

a = []

for i in range(81):
    if columns[i][0][0] == 0 and columns[i][0][1] ==  -1:
        a.append(6)

    elif columns[i][0][0] == 0 and columns[i][0][1] ==  1:
        a.append(2)

    elif columns[i][0][0] == 1 and columns[i][0][1] ==  1:
        a.append(1)

    elif columns[i][0][0] == 1 and columns[i][0][1] ==  0:
        a.append(0)

    elif columns[i][0][0] == 1 and columns[i][0][1] ==  -1:
        a.append(7)

    elif columns[i][0][0] == -1 and columns[i][0][1] ==  1:
        a.append(3)

    elif columns[i][0][0] == -1 and columns[i][0][1] ==  0:
        a.append(4)

    elif columns[i][0][0] == -1 and columns[i][0][1] ==  -1:
        a.append(5)

print(a)
cv.waitKey()
