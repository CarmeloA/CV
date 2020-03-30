import cv2 as cv
import os

img_path = 'E:\zhanglefu\VOCdevkit\VOC2022\JPEGImages\\screwdriver\\'
label_path = 'E:\zhanglefu\VOCdevkit\VOC2022\labels\\screwdriver\\'

for img in os.listdir(img_path):
    image = cv.imread(img_path+img)
    c_image = image.copy()
    height = image.shape[0]
    width = image.shape[1]

    file = open(label_path+img.split('.')[0]+'.txt','r')
    lines = file.readlines()
    for line in lines:
        nums = line.strip().split(' ')
        top_left_x = int(float(nums[1]) * width)
        top_left_y = int(float(nums[2]) * height)
        w = int(float(nums[3]) * width)
        h = int(float(nums[4]) * height)
        cv.rectangle(c_image,(top_left_x,top_left_y),(top_left_x+w,top_left_y+h),(255,0,0),thickness=2)
    cv.imshow('win',c_image)
    cv.waitKey(0)


