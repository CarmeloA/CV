'''
工具类
包含了一些图片预处理的方法
'''
import cv2 as cv
import numpy as np
from PIL import Image
import os

class ItemExtractionTool(object):
    def __init__(self,path='original.png'):
        self.image = cv.imread(path)
        #原图片的宽高
        # self.width = self.image.shape[0]
        # self.height = self.image.shape[1]
        #用来辅助存储第一次截图
        self.img = Image.open(path)
        # 设置蓝色和橙色阈值
        self.lower_blue = np.array([100, 43, 46])
        self.upper_blue = np.array([124, 255, 255])
        self.lower_orange = np.array([11, 43, 46])
        self.upper_orange = np.array([25, 255, 255])
        self.lower_green = np.array([35, 43, 46])
        self.upper_green = np.array([77, 255, 255])
        #设置卷积核用来锐化
        self.kernel_sharpen_3 = np.array([
            [-1, -1, -1, -1, -1],
            [-1, 2, 2, 2, -1],
            [-1, 2, 8, 2, -1],
            [-1, 2, 2, 2, -1],
            [-1, -1, -1, -1, -1]]) / 8.0

    #锐化
    def sharpen(self,image):
        return cv.filter2D(image,-1,self.kernel_sharpen_3)
    #阈值筛选,保留蓝色物体
    def keep_blue_item(self,image):
        hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv,self.lower_blue,self.upper_blue)
        res = cv.bitwise_and(image, image, mask=mask)
        return res
    # 阈值筛选,保留橙色物体
    def keep_orange_item(self, image):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_orange, self.upper_orange)
        res = cv.bitwise_and(image, image, mask=mask)
        return res
    # 阈值筛选,保留绿色物体
    def keep_green_item(self, image):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_green, self.upper_green)
        res = cv.bitwise_and(image, image, mask=mask)
        return res
    #保留红色通道
    def keep_red(self,image):
        red = np.zeros_like(image)
        red[:,:,2] = image[:,:,2]
        return red
    #图像预处理
    def img_pretreatment(self,image):
        #灰度化
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        #降噪
        bilater = cv.bilateralFilter(gray,7,31,31)
        #二值化
        binary = cv.adaptiveThreshold(bilater,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,5)
        return gray,binary
    #边缘检测
    #索贝尔
    def sobel_detection(self,image):
        #sobel
        sobel_x = cv.Sobel(image,cv.CV_64F,1,0,ksize=5)
        sobel_y = cv.Sobel(image,cv.CV_64F,0,1,ksize=5)
        sobel = sobel_x-sobel_y
        return sobel
    def canny_detection(self,image):
        canny = cv.Canny(image,50,240)
        return canny
    #形态学处理
    def structuring(self,image):
        #构造结构元素
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,3))
        #开运算
        # open = cv.morphologyEx(image,cv.MORPH_OPEN,kernel)
        #闭运算
        close = cv.morphologyEx(image,cv.MORPH_CLOSE,kernel)
        open = cv.dilate(image,kernel,iterations=1)
        return open
    #寻找轮廓
    def find_contours(self,image,c_minArea):
        _,contours,hierarchy = cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print('没有找到轮廓!')
            return
        else:
            item_contours = []
            box_list = []
            for i in range(len(contours)):
                #计算所有轮廓的面积,保留较大的轮廓
                area = cv.contourArea(contours[i])
                if area>c_minArea:
                    item_contours.append(contours[i])
                    #计算包围轮廓的有旋转角度的最小矩形面积,用来在原图中标记要识别的物体
                    rect = cv.minAreaRect(contours[i])
                    box = np.int0(cv.boxPoints(rect))
                    box_list.append(box)
                    #返回包围轮廓的最小矩阵的左上角坐标和长、宽
                    x, y, w, h = cv.boundingRect(contours[i])  # 获取轮廓坐标
                    #截取含有目标物的图片
                    item_img = self.image[y:y+h,x:x+w,:]
                    cv.imwrite('item_img.png',item_img)
            return item_contours, box_list,item_img
    #画出轮廓
    def draw_contours(self,item_contours,box_list,image):
        cv.drawContours(image,item_contours,-1,(255,0,0),2)
        cv.drawContours(image,box_list,-1,(0,0,255),2)
        cv.imshow('image_new',image)


    #拆分图片,对只包含待检测物品的截图做拆分,精细每一步操作
    def split_img(self,item_img):
        #创建文件夹用来保存拆分的图片
        # path = 'e:\\zhanglefu\\day05\\img_pieces\\'
        # isExist = os.path.exists(path)
        # if not isExist:
        #     os.makedirs(path)
        #储存碎片的列表
        img_pieces = []
        #原图的宽高
        item_img = item_img[:129,:282,:]
        shape = item_img.shape
        width = shape[1]
        height = shape[0]
        #碎片的宽高
        p_width = width//3
        p_height = height//3
        #设置图片的名称
        w = 1 #行
        for i in range(0,width,p_width):
            if w<4:
                h=1 #列
                for j in range(0, height, p_height):
                    img = item_img[j:j + p_height, i:i + p_width, :]
                    img_pieces.append(img)
                    cv.imwrite('img_pieces/' + str(h) + '-' + str(w) + '.png', img)
                    h+=1
            w+=1
        return img_pieces
    #合并图片
    def merge_img(self,img_pieces):
        # 创建三个列表用于存储水平拼接的三张图片
        h_merge_1 = []
        h_merge_2 = []
        h_merge_3 = []
        # 再创建一个列表存储三张水平合并后的图片
        v_merge = []
        for i in range(len(img_pieces)):
            if i <= 2:
                h_merge_1.append(img_pieces[i])
            elif i <= 5:
                h_merge_2.append(img_pieces[i])
            else:
                h_merge_3.append(img_pieces[i])
        merge_1 = cv.vconcat(h_merge_1)
        merge_2 = cv.vconcat(h_merge_2)
        merge_3 = cv.vconcat(h_merge_3)
        merge_img = cv.hconcat([merge_1, merge_2, merge_3])
        cv.imshow('after_merge',merge_img)

        return merge_img

    #对合成后的小图做轮廓查找
    def find_contours_for_mergeimg(self,image):
        image = np.array(image,np.uint8)
        _,contours,hierarchy = cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print('没有找到轮廓!')
            return
        else:
            item_contours = []
            for i in range(len(contours)):
                #计算所有轮廓的面积,保留较大的轮廓
                area = cv.contourArea(contours[i])
                print(area)
                if area>4000:
                    item_contours.append(contours[i])
        cv.drawContours(cv.imread('item_img.png'),item_contours,-1,(255,0,0),1)
        cv.imshow('image_c',cv.imread('item_img.png'))
        cv.waitKey()
        return item_contours

    #获取一张图片中所有带有颜色的像素点坐标
    def color_points_pos(self,image):
        color_points_list = []
        for i, h in enumerate(image):
            l = []
            for j, w in enumerate(h):
                if 0 not in w:
                    l.append([j, i])
            if len(l):
                color_points_list.append(l)
        # print(color_points_list)
        # print(len(color_points_list))
        lowest_point = color_points_list[-1][-1]
        highest_point = color_points_list[0][0]
        # print(lowest_point)
        # print(highest_point)
        #刀宽
        knife_width = lowest_point[1] - highest_point[1]
        return color_points_list,knife_width

if __name__ == '__main__':
    image = cv.imread('knife.png')
    iet = ItemExtractionTool()
    iet.color_points_pos(image)



