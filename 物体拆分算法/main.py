'''
主程序
用来执行各种算法并做程序扩展
'''
from tools import ItemExtractionTool
from item_segmentation import *
import cv2 as cv
from item_fill import *
from item_mark import item_mark

#图片路径
path = 'original.png'
#创建工具类对象
iet = ItemExtractionTool()
#程序执行主函数
def main(path):
    #按单通道方式读取图片用作锐化处理备用
    original_gray = cv.imread(path,0)
    # 正常读取彩图用作保留单颜色处理备用
    original = cv.imread(path)
    #锐化
    sharpen = iet.sharpen(original_gray)
    #只保留蓝色物体
    blue_item = iet.keep_blue_item(original)
    #区域增长种子
    seeds = [Point(10, 10), Point(82, 150), Point(20, 280)]

    #突出雨伞
    umbrella_binaryImg = regionGrow(sharpen, seeds, 24)
    cv.imwrite('umbrella_binaryImg.png', umbrella_binaryImg * 255)

    #突出刀
    knife = cv.subtract(regionGrow(sharpen, seeds, 19),umbrella_binaryImg)
    white = np.ones_like(knife)
    knife_binaryImg = cv.add(knife,white)
    cv.imwrite('knife_binaryImg.png', knife_binaryImg * 255)

    #与res相乘后再做形态学处理得到目标物体,同时用来获取直线
    knife_binaryImg = cv.imread('knife_binaryImg.png')
    umbrella_binaryImg = cv.imread('umbrella_binaryImg.png')
    #刀
    k = umbrella_binaryImg * blue_item
    k = dilate(k)

    #雨伞
    u = knife_binaryImg * blue_item

    # 获取图中所有直线
    lines = []
    klines = get_line(knife_binaryImg, 75)
    ulines = get_line(umbrella_binaryImg, 75)
    for kl, ul in zip(klines, ulines):
        lines.append(kl)
        lines.append(ul)
    # 获取包含所有交点的列表
    cross_list = get_cross(lines, umbrella_binaryImg)
    # 选取用来连通区域的交点
    use_cross_list = choose_cross(cross_list)
    # 连通区域
    k = patch(k, use_cross_list)

    #在原图标记刀和伞
    knife,umbrella = item_mark(k,u,path)

    #存储最终图片
    cv.imwrite('knife.png', knife)
    cv.imwrite('umbrella.png', umbrella)
    cv.imshow('knife', knife)
    cv.imshow('umbrella', umbrella)
    cv.waitKey()
    return knife,umbrella

if __name__ == '__main__':
    knife,umbrella = main(path)
