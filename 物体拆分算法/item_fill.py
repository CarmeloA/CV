'''
填充断开的物品
利用直线检测获得刀和雨伞的边缘交点，通过填充交点构成的多边形达到补全物品的效果
'''
import cv2 as cv
import numpy as np
from tools import ItemExtractionTool


#创建工具类对象
iet = ItemExtractionTool()

# 获取交线
def get_line(img,threshhold_for_houghlines):
    # img = np.array(img,np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    img = cv.dilate(img,kernel,iterations=1)
    img = cv.erode(img,kernel,iterations=1)
    img = cv.GaussianBlur(img,(3,3),0)
    edges = cv.Canny(img,50,150,apertureSize=3)
    lines = cv.HoughLines(edges,1,np.pi/180,threshhold_for_houghlines)
    # print(lines)
    result = cv.imread('original.png')
    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        # print(rho)
        # print(theta)
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
            # print(pt1)
            # 该直线与最后一行的焦点
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            # print(pt2)
            cv.line(result, pt1, pt2, (255,0,255))  # 绘制一条白线
        else:  # 水平直线
            pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
            # print(pt1)
            # 该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            # print(pt2)
            cv.line(result, pt1, pt2, (255,0,0), 2)  # 绘制一条直线
    #找到截距大于0的直线的索引
    # n = []
    # for i,line in enumerate(lines):
    #     rho = line[0][0]  # 第一个元素是距离rho
    #     if rho>0:
    #         n.append(i)
    # cv.imshow('Result', result)
    # cv.waitKey()
    return lines

#计算交点
def get_cross_formula(r1,t1,r2,t2):
    x = (r2/np.sin(t2)-r1/np.sin(t1))/(np.cos(t2)/np.sin(t2)-np.cos(t1)/np.sin(t1))
    y = x*(-np.cos(t1)/np.sin(t1))+(r1/np.sin(t1))
    return x, y

def get_cross(lines,image):
    cross_list = []
    for i in range(len(lines)):
        #存储每条直线和其他直线的交点
        c_list = []
        for j in range(len(lines)):
            if not(lines[i] is lines[j]):
                x,y = get_cross_formula(lines[i][0][0],lines[i][0][1],lines[j][0][0],lines[j][0][1])
                if (x<=image.shape[0] and x>=0) and (y<=image.shape[1] and y>=0):
                    c_list.append((round(x,3),round(y,3)))
        cross_list.append(c_list)
    return cross_list

#选取连通区域的交点
def choose_cross(cross_list):
    all_cross_list = []
    for C in cross_list:
        for c in C:
            all_cross_list.append(c)
    all_cross_list = list(set(all_cross_list))
    use_cross_list = sorted(all_cross_list,key=lambda x:x[0])
    use_cross_list[-1],use_cross_list[-2] = use_cross_list[-2],use_cross_list[-1]
    return use_cross_list
# 绘制连通区域
def patch(image,use_cross_list):
    cv.fillPoly(image,np.array([use_cross_list],dtype=np.int32),[0,255,0])
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # image = cv.dilate(image, kernel, iterations=1)
    return image

if __name__ == '__main__':
    kbi = cv.imread('knife_binaryImg.png')
    ubi = cv.imread('umbrella_binaryImg.png')
    lines = []
    klines = get_line(kbi)
    ulines = get_line(ubi)
    for kl,ul in zip(klines,ulines):
        lines.append(kl)
        lines.append(ul)
    print(lines)
    cross_list = get_cross(lines,kbi)
    print(cross_list)
    use_cross_list = choose_cross(cross_list)
    print(use_cross_list)
