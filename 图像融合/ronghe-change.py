import os
import shutil
import time
import cv2
import numpy as np
import random
import sys
import getopt
import argparse
import matplotlib.pyplot as plt
# from .ronghe_txt2xml import get_path, separate, get_xml
# 读取图片，使枪械图片变蓝
def img_blue(img_path):
    img = cv2.imread(img_path)
    print()
    # img2 = np.zeros((608,608,3), dtype = 'uint8')*256
    # img = cv2.resize(img,(608,416))
    # print(img.shape)
    b, g, r = cv2.split(img)
    b[b < 210] = 200
    g[g < 210] = 0
    r[r < 210] = 0
    img = cv2.merge([b, g, r])
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('E:/Faster-RCNN_TF/new-work/gun/' + img_path.split('/')[-1], img)

    return img, img_path.split('/')[-1]
# 读取图片文件夹
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1][1:] in suffix:
                L.append(os.path.join(root, file))
    return L

# resize操作
def img_resize(img, name, save_dir1):
    im = img[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    save_dir = 'D:/Cai-work/animal(7.19)/plant-resize/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # plt.savefig(os.path.join(save_dir, name))
    plt.close()
    return save_dir+name

# 找到物体轮廓
def find_outline(image_path, n=None, flag1=False, flag2=False):
    img = cv2.imread(image_path)
    print('image_path:', image_path)
    sp = img.shape
    height = sp[0]  # height(rows) of image
    width = sp[1]  # width(colums) of image
    # print('weight', width)
    # print('height', height)

    if flag1 == True:
        # img = cv2.resize(img, (int(608*1.5/3), int(608*1.5/3)), interpolation=cv2.INTER_CUBIC)
        # img = cv2.resize(img, (int(608/5), int(608/5)), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # if flag2==True:
    #     gray[gray>190]=255
    # 用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
    # 获得高水平梯度和低垂直梯度的图像区域
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow("gradient", gradient)

    # 去除噪声
    blurred = cv2.blur(gradient, (5, 5))  # 平均平滑
    (_, thresh) = cv2.threshold(blurred , 100, 255, cv2.THRESH_BINARY)
    if flag2 == True:
        thresh = cv2.dilate(thresh, None, iterations=1)
    # cv2.imshow("thresh", thresh)
    # image1,cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)) # 构建特定大小的结构性元素
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # 先膨胀再腐蚀。被用来填充前景物体中的小洞，或者前景上的小黑点
    # cv2.imshow("closed", closed)
    # #腐蚀与膨胀
    closed = cv2.erode(closed, None, iterations=2) # 去除图像上的白噪音
    closed = cv2.dilate(closed, None, iterations=2) # 还原图像，但白噪音不会还原
    # 绘制包所在位置的矩形边框
    image1, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cs = sorted(cnts, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cs[0])
    # cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
    # cv2.imshow("find_outline",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(image_path)
    # print(w)
    # print(h)
    return (img, x, y, x + w, y + h)


# 加深或变浅刀具颜色
def un_deepcolor_image(src1, flag2, n):
    added2 = []
    M = np.ones(src1.shape, dtype="uint8") * n
    # cv2.imshow('m',M)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if flag2 == 0:  # 颜色加深的刀
        added2 = cv2.subtract(src1, M)
        # cv2.imshow('add2',added2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        added2[added2 >= 150] = 255
        # cv2.imshow('added2', added2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print('added2', added2)
    elif flag2 == 1:  # 颜色变浅的刀
        added2 = cv2.add(src1, M)
    return added2


# 粗略的调节对比度和亮度
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    src2 = np.ones([h, w, ch], src1.dtype) * 255
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    # cv2.imshow("con-bri-demo", dst)
    return dst


# 转换yolo坐标
def convert(size, box):
    # print("box:", box)
    dw = 1. / size[0]
    dh = 1. / size[1]
    xs = (box[0] + box[1]) / 2.0
    ys = (box[2] + box[3]) / 2.0
    wx = box[1] - box[0]
    hy = box[3] - box[2]
    # print(xs, ys, wx, hy)
    xs = xs * dw
    wx = wx * dw
    ys = ys * dh
    hy = hy * dh
    return (xs, ys, wx, hy)


# 保存图片
# def savefile(image, xlist, file_dir, width, height, bag_name):
#     for location_value in xlist:
#         cl = location_value[0]
#         xmin = location_value[1]
#         xmax = location_value[3]
#         ymin = location_value[2]
#         ymax = location_value[4]
#         b = (float(xmin), float(xmax), float(ymin), float(ymax))
#         bb = convert((width, height), b)
#         # save_txt = os.path.join(file_dir, str(num) + ".txt")
#         save_txt = os.path.join(file_dir, bag_name + ".txt")
#         # with open(save_txt, "a") as f1:
#         #     f1.write(str(cl) + " " + " ".join([str(a) for a in bb]) + "\r")
#         # save_image = os.path.join(file_dir, str(num) + ".jpg")
#         save_image = os.path.join(file_dir, bag_name + ".jpg")
#         # cv2.imwrite(save_image, image)
#         cv2.imshow('ronghe', image)
#         a = cv2.waitKey()
#         # print('a', a)
#         if a == 32:
#             # os.remove(save_image)
#             # os.remove(save_txt)
#             # os.remove(save_txt2)
#             # os.remove(save_image2)
#             cv2.imwrite(save_image, image)
#             with open(save_txt, "a") as f1:
#                 f1.write(str(cl) + " " + " ".join([str(a) for a in bb]) + "\r")
#         cv2.destroyAllWindows()
def savefile(image, xlist, file_dir, width, height, bag_name):
    for location_value in xlist:
        cl = location_value[0]
        xmin = location_value[1]
        xmax = location_value[3]
        ymin = location_value[2]
        ymax = location_value[4]
        b = (float(xmin), float(xmax), float(ymin), float(ymax))
        bb = convert((width, height), b)
        save_txt = os.path.join(file_dir, str(num) + ".txt")

        save_image = os.path.join(file_dir, str(num) + ".jpg")
        cv2.imshow('gem', image)
        a = cv2.waitKey(0)
        if a == 32:
            cv2.imwrite(save_image, image)
            with open(save_txt, "a+") as f1:
                f1.write(str(cl) + " " + " ".join([str(a) for a in bb]) + "\r\n")
                f1.closed
            # pass
        cv2.destroyAllWindows()

# 从终端获取参数
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='daruonghe')
  parser.add_argument('--gun', dest='gun_path',
                      help='the gun path',
                      default='D:/Cai-work/animal(7.19)/knife/', type=str)
  parser.add_argument('--bag', dest='bag_path',
                      help='the bag path',
                      default='D:/Cai-work/animal(7.19)/bag/',
                      type=str)
  parser.add_argument('--save', dest='save_path',
                      help='the save path',
                      default='D:/Cai-work/animal(7.19)/bag-knife/', type=str)
  parser.add_argument('--cls', dest='c',
                      help='the number of class',
                      default=34, type=int)
  parser.add_argument('--narrow', dest='s',
                      help='x光枪缩小的比例',
                      default=0.5, type=float)
  parser.add_argument('--addorsub_flag', dest='f',
                      help='x抢图颜色加深或者变浅标志位',
                      default=0, type=int)
  parser.add_argument('--un_deepth', dest='d',
                      help='变浅或者加深数值',
                      default=90, type=int)
  parser.add_argument('--weight', dest='w',
                      help='融合权重', default=0.7,
                      type=float)
  parser.add_argument('--blacklight_ratio', dest='r',
                      help='明暗比',
                      default=1.7, type=float)

  args = parser.parse_args()
  return args

num = 0  # 图片数量
start = time.time()

xknife_folder = ""
xbag_folder = ""
save_path = ""
cls = None
addorsub_flag = None  # 加深或变浅标志位
narrow = None  # 刀缩小比列
un_deepth = None  # 加深或变浅数值
weight = None  # 融合权重
blacklight_ratio = None  # 明暗的比列

args = parse_args()

cls = args.c
narrow = args.s
addorsub_flag = args.f
un_deepth = args.d
weight = args.w
blacklight_ratio = args.r

xknife_folder = args.gun_path
xbag_folder = args.bag_path
save_path = args.save_path


# if image_path[-1]!="/":
#     image_path1 = image_path+"/"
# if save_path[-1]!="/":
#     copy_path = copy_path+"/"
if os.path.exists(save_path):  # 判断是否存在此文件夹
    pass
else:
    os.makedirs(save_path)

suffix = ["png", "bmp", "jpg", "BMP"]
sufftxt = ["txt"]
num_list = []
index1 = 0
save_pathlist = os.listdir(save_path)

# 找出要存储文件夹中 num最大
if save_pathlist == []:
    num = 0
else:
    for img_name_str in save_pathlist:
        if os.path.isfile(os.path.join(save_path, img_name_str)):
            filesuffix1 = os.path.splitext(img_name_str)[1][1:]
            # print("filesuffix1:",filesuffix1)
            if filesuffix1 in suffix:
                num_images = img_name_str.split(".")[0] # 得到的是一个图片名
                for num_img in num_images: #例如：num_images = 548
                    if num_img != "0":
                        index1 = num_images.index(num_img)
                        # print(index1)
                        break
                num = num_images[index1:]
                num_list.append(int(num))
    num = max(num_list) + 1

# 读取图片文件夹
for bag_image in file_name(xbag_folder):
    # bag_name = os.path.splitext(bag_image)[0].split('/')[-1]
    # with open(xbag_folder + bag_name + '.txt') as tf:
    #     xlist = tf.readlines()
    #     print(len(xlist))
    #     if len(xlist) == 2:
    # if int(bag_name) <= 779:


            for knife_image in file_name(xknife_folder):
                print('knife_name: ', knife_image)
                knife_image=random.choice(file_name(xknife_folder)) # 随  机选取一张图片
                # print("knife_image:", knife_image)

                # 图片变蓝操作
                # img, name = img_blue(knife_image)
                img = cv2.imread(knife_image)
                # img = cv2.blur(img, (5,5))
                name = knife_image.split('/')[-1]
                # knife_image = img_resize(img, name, xknife_folder)
                # print('knife_inage', knife_image)
                filesuffix = os.path.splitext(bag_image)[1][1:]
                bag_name = os.path.splitext(bag_image)[0].split('/')[-1]

                # 接着上一天未完成的工作进行
                # if int(bag_name) >= 3186:
                print('bag_name:', bag_name)

                # print(filesuffix)
                if filesuffix in suffix:
                    f_list = []
                    numpy_bimg, bx_s, by_s, bx_e, by_e = find_outline(bag_image)
                    # print(numpy_bimg, bx_s, by_s, bx_e, by_e)
                    sp = numpy_bimg.shape
                    he = sp[0]  # height(rows) of image
                    wi = sp[1]  # width(colums) of image
                    # print("bw:",bx_e-bx_s)
                    # print("bh:",by_e-by_s)
                    numpy_kimg, kx_s, ky_s, kx_e, ky_e = find_outline(knife_image, n=float(narrow), flag1=True, flag2=True)
                    save_txt2 = os.path.join('D:/Cai-work/animal(7.19)/plant2-resize/', knife_image.split('/')[-1].split('.')[0] + ".txt")
                    save_image2 = os.path.join('D:/Cai-work/animal(7.19)/plant2-resize/', knife_image.split('/')[-1].split('.')[0] + ".jpg")
                    with open(save_txt2, "w") as f2:
                        h, w, c = img.shape
                        # print(img.shape)
                        b = (float(kx_s*(1/narrow)), float(kx_e*(1/narrow)), float(ky_s*(1/narrow)), float(ky_e*(1/narrow)))
                        bb = convert((w, h), b)
                        f2.write(str(cls) + " " + " ".join([str(a) for a in bb]) + "\r\n")
                        cv2.imwrite(save_image2, img)
                        f2.closed
                    # #融合图片
                    image3 = numpy_kimg[ky_s:ky_e, kx_s:kx_e]  # 从裸刀的x光图下取出

                    # if image3.shape[0] >=100:
                    #     image3 = cv2.resize(image3,(99,119))
                    #
                    # cv2.imshow('image3', image3)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # print(kx_e, ky_e, kx_e - kx_s, ky_e - ky_s)
                    # if addorsub_flag  == "0":  ##颜色加深的刀
                    #     image3 = deepcolor_image(image3,50)
                    # if addorsub_flag  =="1": #颜色变浅的刀
                    #     image3 = undeepcolor_image(image3,20)
                    # 刀具颜色加深
                    # print(un_deepth)
                    # print(addorsub_flag)
                    image3 = un_deepcolor_image(image3, addorsub_flag, int(un_deepth))
                    #print('image3', image3)
                    # cv2.imshow("image3", image3)#把刀部分拿出来
                    # cv2.imwrite("C:\\Users\\Administrator\\Desktop\\12\\5.jpg" ,image3)
                    # 假如包w是160，h是70，刀w40 h20

                    ############change by cai
                    xs_limit = int((bx_e - bx_s) / 8) #包w/8   --20
                    xe_limit = int(7 * (bx_e - bx_s) / 8) - (kx_e - kx_s) #7*包w/8-刀w   --100

                    # xs_limit = int((bx_e - bx_s) / 2) - 3*(kx_e - kx_s)
                    # xe_limit = int((bx_e - bx_s) / 2) + 3*(kx_e - kx_s)
                    if xs_limit < xe_limit:
                        w_random = random.randint(xs_limit, xe_limit)  # 随机一个w偏移量
                        x_s = bx_s + w_random #刀在包中的左上角x坐标  --最小包左x+20，最大包左x+100
                        x_e = x_s + (kx_e - kx_s) #刀在包中的最右边坐标   --最小包左x+60，最大包左x+140
                        ys_limit = int((bx_e - bx_s) / 8)  # 包w/8 --20
                        ye_limit = int(7 * (by_e - by_s) / 8) - (ky_e - ky_s)  # 7*包h/8-刀h   --41.25
                        # ys_limit = int((by_e - by_s) / 2) - 3*(ky_e - ky_s)
                        # ye_limit = int((by_e - by_s) / 2) + 3*(ky_e - ky_s)
                        # print(ys_limit)
                        # print(ye_limit)
                        if ys_limit < ye_limit:
                            h_random = random.randint(ys_limit, ye_limit)  # 随机一个h偏移量
                            # print(h_random)
                            y_s = by_s + h_random  # 刀在包中上边y坐标  --最小包y+20，最大包y+41.25
                            y_e = y_s + (ky_e - ky_s)  # 刀在包中下边y坐标   --最小为包y+40，最大为包y+61.25
                            roi = numpy_bimg[y_s:y_e, x_s:x_e]  # 从x包图上随机切下同样大小的一块
                            # cv2.imshow("roi1:",roi)#从要放入的图中拿出这部分
                            # cv2.imwrite("C:\\Users\\Administrator\\Desktop\\12\\4.jpg" ,roi)

                            result = cv2.addWeighted(roi, 0.7, image3, 0.3, 0)  # roi与原刀具图像融合，roi与images必须尺寸相同dst = src1 * alpha + src2 * beta + gamma;
                            # cv2.imshow("result",result)
                            result = contrast_brightness_image(result, 1.2, 1)  # 粗略的调节对比度和亮度
                            numpy_bimg[y_s:y_e, x_s:x_e] = result
                            # cv2.imshow("numpy_bimg:",numpy_bimg)

                            M1 = np.ones(image3.shape, dtype="uint8") * 100  # 与刀具image大小一样的全100矩阵
                            added = cv2.add(image3, M1)  # 将图像image与M1相加
                            # cv2.imshow("added",added)
                            w = float(weight)
                            # result3=cv2.addWeighted(roi,0.7,added,1-0.7,0)
                            # result3=cv2.addWeighted(roi,0.5,added,1-0.5,0)
                            # cv2.imshow("roi2:",roi)
                            result3 = cv2.addWeighted(roi, w, added, 1 - w, 0) #r oi与处理后的刀具图融合
                            # cv2.imshow("result3",result3)
                            ratio = float(blacklight_ratio) # 明暗比
                            # result3=contrast_brightness_image(result3, 1.7, 1)#
                            # result3=contrast_brightness_image(result3, 2.5, 1)
                            result3 = contrast_brightness_image(result3, ratio, 1)
                            numpy_bimg[y_s:y_e, x_s:x_e] = result3
                            # cv2.imshow("numpy_bimg2:",numpy_bimg)
                            f_list.append((cls, x_s, y_s, x_e, y_e))

                            # savefile(numpy_bimg, f_list, save_path, wi, he, save_txt2, save_image2, bag_name)
                            savefile(numpy_bimg, f_list, save_path, wi, he, bag_name)
                            num += 1
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

end = time.time()
print("num size:", num)
print("total_time：", end - start)