import os
import shutil
import time
import cv2
import numpy as np
import random
import sys
import getopt

# 读取图片文件夹
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1][1:] in suffix:
                L.append(os.path.join(root, file))
    return L

#找到物体轮廓
def find_outline(image_path,n=None,flag1=False,flag2=False):
    img = cv2.imread(image_path)
    print(type(img))
    sp = img.shape
    height = sp[0]  # height(rows) of image
    width = sp[1]  # width(colums) of image
    if flag1==True:
        # img = cv2.resize(img, (int(608*1.5/3), int(608*1.5/3)), interpolation=cv2.INTER_CUBIC)
        # img = cv2.resize(img, (int(608/5), int(608/5)), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (int(width*n), int(height*n)), interpolation=cv2.INTER_CUBIC)  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # if flag2==True:
    #     gray[gray>190]=255
    #用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
    #获得高水平梯度和低垂直梯度的图像区域
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow("gradient", gradient)

    # 去除噪声
    blurred = cv2.blur(gradient, (5, 5))
    (_, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    if flag2==True:
        thresh=cv2.dilate(thresh, None, iterations=1)
    # cv2.imshow("thresh", thresh)
    # image1,cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed", closed)
    # #腐蚀与膨胀
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)
    #绘制包所在位置的矩形边框
    image1,cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cs = sorted(cnts, key=cv2.contourArea, reverse=True)                
    x,y,w,h = cv2.boundingRect(cs[0])
    # cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
    # cv2.imshow("find_outline",img)
    return (img,x,y,x+w, y+h)   

#加深或变浅刀具颜色
def un_deepcolor_image(src1,flag2,n):
    M = np.ones(src1.shape,dtype="uint8")*n
    if flag2 =="0": #颜色加深的刀
        added = cv2.subtract(src1,M)
        added[added>=150]=255
    elif flag2 =="1": #颜色变浅的刀 
        added = cv2.add(src1,M)
    return added


#粗略的调节对比度和亮度
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape#获取shape的数值，height和width、通道
    src2 = np.ones([h, w, ch], src1.dtype)*255
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)#addWeighted函数说明如下 
    # cv2.imshow("con-bri-demo", dst) 
    return dst

# 转换yolo坐标
def convert(size, box):
    print("box:", box)
    dw = 1./size[0]
    dh = 1./size[1]
    xs = (box[0] + box[1])/2.0
    ys = (box[2] + box[3])/2.0
    wx = box[1] - box[0]
    hy = box[3] - box[2]
    print(xs, ys, wx, hy)
    xs = xs*dw
    wx = wx*dw
    ys = ys*dh
    hy = hy*dh
    return (xs, ys, wx, hy)

#保存图片
def savefile(image, xlist, file_dir, width, height):
    for location_value in xlist:
        cl = location_value[0] 
        xmin = location_value[1]
        xmax = location_value[3]
        ymin = location_value[2]
        ymax = location_value[4]
        b = (float(xmin), float(xmax), float(ymin), float(ymax))
        bb = convert((width, height), b) 
        save_txt = os.path.join(file_dir,str(num)+".txt")
        with open(save_txt, "a+") as f1:
            f1.write(str(cl) + " " + " ".join([str(a) for a in bb])+"\r\n")
        save_image = os.path.join(file_dir,str(num)+".jpg")
        cv2.imshow('res',save_image)
        cv2.waitKey(0)
        cv2.imwrite(save_image ,image)

num =0 #图片数量
start = time.time()


xknife_folder = ""
xbag_folder = ""
save_path = ""
cls=""
addorsub_flag = "" #加深或变浅标志位
narrow = "" #刀缩小比列
un_deepth = ""#加深或变浅数值
weight = ""  #融合权重
blacklight_ratio = ""#明暗的比列


opts, args = getopt.getopt(sys.argv[1:], 'c:s:f:d:w:r:')

if len(args) < 3 and len(opts) < 6:
    sys.stderr.write("help: \n")
    sys.stderr.write(" composite_pic.py [-c -s -f -d -w -r] xknife_folder xbag_folder save_path\n")
    sys.exit(-1)

xknife_folder = args[0]
xbag_folder = args[1]
save_path = args[2]


for papar,value in opts:
    if papar == "-c":
        cls = value
    elif papar == "-s":
        narrow = value
    elif papar == "-f":
        addorsub_flag = value
    elif papar == "-d":
        un_deepth = value
    elif papar == "-w":
        weight = value
    elif papar == "-r":
        blacklight_ratio = value


# xknife_folder = "C:\\Users\\Administrator\\Desktop\\knife"
# xbag_folder = "C:\\Users\\Administrator\\Desktop\\bag1"
# save_path = "C:\\Users\\Administrator\\Desktop\\ronghe"


# if image_path[-1]!="/":
#     image_path1 = image_path+"/" 
# if save_path[-1]!="/":
#     copy_path = copy_path+"/"
if os.path.exists(save_path): #判断是否存在此文件夹
    pass
else:
    os.makedirs(save_path)

suffix = ["png","bmp","jpg","BMP"]
sufftxt =["txt"] 
num_list=[]
index1=0
save_pathlist = os.listdir(save_path)
print(save_pathlist)

#找出要存储文件夹中 num最大
if save_pathlist == []:
    num=0
else:   
    for img_name_str in save_pathlist:
        if os.path.isfile(os.path.join(save_path,img_name_str)):
            filesuffix1 = os.path.splitext(img_name_str)[1][1:]
            # print("filesuffix1:",filesuffix1)
            if filesuffix1 in suffix:
                num_images = img_name_str.split(".")[0]
                for num_img in num_images:
                    if num_img != "0":
                        index1 = num_images.index(num_img)
                        # print(index1)
                        break
                num = num_images[index1:]
                num_list.append(int(num))
    num = max(num_list)+1


#读取图片文件夹
for bag_image in file_name(xbag_folder):
    for knife_image in file_name(xknife_folder):
    # knife_image=random.choice(file_name(xknife_folder)) # 随机选取一张图片
        print("knife_image:",knife_image)
        filesuffix = os.path.splitext(bag_image)[1][1:]
        # print(filesuffix)
        if filesuffix in suffix:
            f_list=[]
            numpy_bimg,bx_s,by_s,bx_e,by_e = find_outline(bag_image)
            sp = numpy_bimg.shape
            he = sp[0]  # height(rows) of image
            wi = sp[1]  # width(colums) of image
            # print("bw:",bx_e-bx_s)
            # print("bh:",by_e-by_s)
            numpy_kimg,kx_s,ky_s,kx_e,ky_e = find_outline(knife_image,n=float(narrow),flag1=True,flag2=True)
            # #融合图片  
            image3 = numpy_kimg[ky_s:ky_e,kx_s:kx_e]  #从裸刀的x光图下取出
            print(kx_e,ky_e,kx_e-kx_s,ky_e-ky_s)
            # if addorsub_flag  == "0":  ##颜色加深的刀
            #     image3 = deepcolor_image(image3,50)   
            # if addorsub_flag  =="1": #颜色变浅的刀
            #     image3 = undeepcolor_image(image3,20)
            image3 = un_deepcolor_image(image3,addorsub_flag,int(un_deepth))
            # cv2.imshow("image3", image3)#把刀部分拿出来
            # cv2.imwrite("C:\\Users\\Administrator\\Desktop\\12\\5.jpg" ,image3)
            xs_limit = int((bx_e-bx_s)/8)
            xe_limit = int(7*(bx_e-bx_s)/8)-(kx_e-kx_s)
            w_random = random.randint(xs_limit,xe_limit)
            x_s = bx_s+w_random
            x_e = x_s+(kx_e-kx_s)
            ys_limit=int((bx_e-bx_s)/8)
            ye_limit = int(7*(by_e-by_s)/8)-(ky_e-ky_s)
            h_random=random.randint(ys_limit,ye_limit)
            y_s =by_s+h_random
            y_e = y_s+(ky_e-ky_s)
            roi=numpy_bimg[y_s:y_e,x_s:x_e] #从x包图上切下同样大小的一块 
            # cv2.imshow("roi1:",roi)#从要放入的图中拿出这部分
            # cv2.imwrite("C:\\Users\\Administrator\\Desktop\\12\\4.jpg" ,roi)
            
            result=cv2.addWeighted(roi,0.7,image3,0.3,0)
            # cv2.imshow("result",result)
            result=contrast_brightness_image(result, 1.2, 1)
            numpy_bimg[y_s:y_e,x_s:x_e] = result
            # cv2.imshow("numpy_bimg:",numpy_bimg)

            M1 = np.ones(image3.shape,dtype="uint8")*100  #与image大小一样的全100矩阵
            added = cv2.add(image3,M1)    #将图像image与M1相加
            # cv2.imshow("added",added)
            w = float(weight)
            # result3=cv2.addWeighted(roi,0.7,added,1-0.7,0)
            # result3=cv2.addWeighted(roi,0.5,added,1-0.5,0)
            # cv2.imshow("roi2:",roi)
            result3=cv2.addWeighted(roi,w,added,1-w,0)
            # cv2.imshow("result3",result3)
            ratio = float(blacklight_ratio)
            # result3=contrast_brightness_image(result3, 1.7, 1)#
            # result3=contrast_brightness_image(result3, 2.5, 1)
            result3=contrast_brightness_image(result3, ratio, 1)
            numpy_bimg[y_s:y_e,x_s:x_e] = result3
            # cv2.imshow("numpy_bimg2:",numpy_bimg)
            f_list.append((cls, x_s, y_s, x_e, y_e))

            savefile(numpy_bimg,f_list, save_path,wi, he)
            num+=1
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

end = time.time()
print("num size:", num)
print("total_time：", end-start)