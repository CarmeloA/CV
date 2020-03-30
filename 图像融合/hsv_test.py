'''
图像融合算法模块
'''
import cv2 as cv
import numpy as np

# 刀具图像预处理并制作其二值化图像
def knife_pretreatment(img, threshold):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 去噪
    blurred = cv.blur(gray, (5, 5))
    # 二值化制作模板
    _, thresh = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY_INV)
    image1, cnts, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return img,thresh,0,0,0,0
    cs = sorted(cnts, key=cv.contourArea, reverse=True)
    x, y, w, h = cv.boundingRect(cs[0])
    # 修改原图,只保留刀具部分
    cv.imshow('knife', img)
    cv.imshow('knife_thresh', thresh)
    img = img[y:y + h, x:x + w]
    # 修改模板,保留和图片大小一致的部分
    thresh = cv.cvtColor(thresh,cv.COLOR_GRAY2BGR)
    thresh = thresh[y:y + h, x:x + w]
    thresh = cv.cvtColor(thresh,cv.COLOR_BGR2GRAY)
    # 截取部分的长宽比
    ratio = h/w
    return img,thresh,x,y,w,h,ratio

# 制作反向模板
def knife_pretreatment_inv(img,threshold):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 去噪
    blurred = cv.blur(gray, (5, 5))
    # 二值化制作模板
    _, thresh = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY)
    cv.imshow('inv',thresh)
    return thresh

# 背景预处理
def background_pretreatment(img,threshold,ksize=(3,3)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.blur(gray, (5, 5))
    _, thresh = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=ksize)
    closed = cv.erode(thresh, kernel, iterations=1)
    closed = cv.dilate(closed, kernel, iterations=1)
    return closed

# 归一化函数
def maxminnorm(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_rows = array.shape[0]
    data_cols = array.shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

'''
融合算法
h:取该位置下,刀具和背景图h的较大值
s:取对应h的s值
v:v_max-v_min*((h_min+1)/(h_min+h_max+1))
'''
def func1(k_h, k_s, k_v, b_h, b_s, b_v,bg_thresh_inv,n):
    new_mh = k_h.copy()
    new_ms = k_s.copy()
    new_mv = k_v.copy()
    height = k_h.shape[0]
    width = k_h.shape[1]
    for i in range(height):
        for j in range(width):
            v_max = max(k_v[i][j], b_v[i][j])
            v_min = min(k_v[i][j], b_v[i][j])
            h_max = max(k_h[i][j], b_h[i][j])
            h_min = min(k_h[i][j], b_h[i][j])
            # new_mv[i][j] = v_max-v_min*((h_min+1)/(h_min+h_max+1))

            # 挑出背景是0的部分,直接使用前景
            if bg_thresh_inv[i][j] == 0:
                new_mh[i][j] = k_h[i][j]
                new_ms[i][j] = k_s[i][j]
                new_mv[i][j] = k_v[i][j]
                # print(i,j,b_v[i][j])
            else:
                # 前景为0用背景
                # TODO:v的取值 叠加后应取较小值即较暗的值
                if k_v[i][j] <= b_v[i][j]:
                    if k_h[i][j] >= b_h[i][j]:
                        new_mv[i][j] = k_v[i][j] - (v_min * pow((h_min + 1) / (h_min + h_max + 1),2))
                        new_mh[i][j] = k_h[i][j]
                        new_ms[i][j] = k_s[i][j]


                    else:
                        new_mv[i][j] = k_v[i][j] - (v_min * pow((h_min + 1) / (h_min + h_max + 1), 2))
                        if n == 11 or n == 15:
                            new_mh[i][j] = k_h[i][j]
                        else:
                            new_mh[i][j] = b_h[i][j]
                        new_ms[i][j] = b_s[i][j]


                else:
                    if k_h[i][j] >= b_h[i][j]:
                        new_mv[i][j] = b_v[i][j] - (v_min * pow((h_min + 1) / (h_min + h_max + 1), 2))
                        new_mh[i][j] = k_h[i][j]
                        new_ms[i][j] = k_s[i][j]


                    else:
                        new_mv[i][j] = b_v[i][j] - (v_min * pow((h_min + 1) / (h_min + h_max + 1), 2))
                        if n == 11 or n == 15:
                            new_mh[i][j] = k_h[i][j]
                        else:
                            new_mh[i][j] = b_h[i][j]
                        new_ms[i][j] = b_s[i][j]



    new_m_hsv = cv.merge([new_mh, new_ms, new_mv])
    new_m_bgr = cv.cvtColor(new_m_hsv, cv.COLOR_HSV2BGR)

    return new_mh, new_ms, new_mv, new_m_hsv, new_m_bgr

# 图像旋转和镜像(默认不镜像)
def rotation_and_flip(img,angle,flip_style,scale=1.0,flag=False):
    # flip_style 1 水平镜像 0 垂直镜像 -1 对角镜像
    if flag:
        img = cv.flip(img,flip_style,dst=None)
    height = img.shape[0]
    width = img.shape[1]
    (cX, cY) = (width // 2, height // 2)
    M = cv.getRotationMatrix2D((cX, cY), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv.warpAffine(img, M, (nW, nH))

