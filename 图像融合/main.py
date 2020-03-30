'''
主程序
'''
from hsv_test import *
from image_fusion_function import *
import argparse
import time
import random

# 物体-类别编号字典
ITEM_DICT = {0:'umbrella',1:'pliers',3:'cellphone',4:'laptop',5:'watch',6:'keys',
             8:'cup',11:'bottle',13:'screwdriver',14:'spoon_fork',15:'lighter',
             16:'bigknife',17:'smallknife',18:'unnormalknife',19:'knife'}

# 一张图片中的物体（填ITEM_DICT中的键）
L = [15]

def parse_args():
    # 当前主要物体
    item = 'lighter'

    parser = argparse.ArgumentParser(description='daruonghe')
    # 物体图路径
    parser.add_argument('--item', dest='item_path',
                        help='the item path',
                        default='E:\\zhanglefu\\dl_img\\{}\\'.format(item), type=str)
    # 背景图路径
    parser.add_argument('--bg', dest='bg_path',
                        help='the bg path',
                        default='E:\\zhanglefu\\bg\\STLX6550_bg\\',
                        type=str)
    # 合成图保存路径
    parser.add_argument('--save_img', dest='img_save_path',
                        help='the save path',
                        default='E:\zhanglefu\VOCdevkit\VOC2022\JPEGImages\\{}\\'.format(item), type=str)
    # 标记框txt保存路径
    parser.add_argument('--save_label', dest='label_save_path',
                        help='the save path',
                        default='E:\zhanglefu\VOCdevkit\VOC2022\labels\\{}\\'.format(item), type=str)

    # 类别名称
    parser.add_argument('--cls_name', dest='c_name',help='the name of class',default='{}-'.format(item), type=str)
    # 随机抽取背景图片的数量
    parser.add_argument('--r_bg_num',dest='r_bg_num',default=40,type=int)
    # 程序执行的round
    parser.add_argument('--round',dest='round',default=20,type=int)

    # 刀具二值化模板阈值
    parser.add_argument('--knife_threshold',dest='knife_threshold',default=230,type=int)
    # 刀具反向二值化模板阈值
    parser.add_argument('--knife_inv_threshold', dest='knife_inv_threshold', default=230, type=int)
    # 背景模板阈值
    parser.add_argument('--bg_inv_threshold',dest='bg_inv_threshold',default=170,type=int)

    # 背景图resize大小
    parser.add_argument('--bg_r_width', dest='bg_r_width', default=608, type=int)
    parser.add_argument('--bg_r_height', dest='bg_r_height', default=608, type=int)

    # 合成图片的总张数
    parser.add_argument('--total_img', dest='total_img', default=100, type=int)
    args = parser.parse_args()
    return args




def main():
    NUM_OF_USEFUL_PIC = 0
    args = parse_args()
    for i in range(args.round):
        # print('共{}张背景,正使用第{}张'.format(str(len(background_list)), str(i+1)))
        print('第{}轮开始...'.format(str(i+1)))
        round_start_time = time.time()
        for j in range(args.r_bg_num):
            background = random.choice(file_name(args.bg_path))
            background_original = cv.imread(background)
            # 修改图片尺寸
            background_original = cv.resize(background_original,(args.bg_r_width,args.bg_r_height))

            # 背景原图副本
            background_c = background_original.copy()
            background_width = background_original.shape[1]
            background_height = background_original.shape[0]
            t = get_background_roi(background_original)
            cv.rectangle(background_original,(t[0],t[1]),(t[0]+t[2],t[1]+t[3]),(255,255,0),thickness=2)
            print((t[0],t[1]),(t[0]+t[2],t[1]+t[3]))
            if t == (0, 0, 0, 0):
                continue
            flag = False
            for n in L:
                item = ITEM_DICT[n]
                pic_path = 'E:\\zhanglefu\\dl_img\\{}\\'.format(item)
                print('选用的物体是{},物体图片路径为:{}'.format(item,pic_path))

                knife = random.choice(file_name(pic_path))
                # print('第{}张图片{}+{}开始合成...'.format(str(j + 1), knife, background))
                knife_original = cv.imread(knife)
                # 正向二值化图像——呈现刀具区域
                knife, knife_thresh, posx, posy, w, h, r = knife_pretreatment(knife_original,
                                                                              threshold=args.knife_threshold)
                print('knife-thresh:', knife_thresh.shape)
                if w == 0 and h == 0:
                    # print('第{}图片无法合成'.format(str(j + 1)))
                    continue
                # 反向二值化图像,用来在背景图中找到放置刀具的位置
                knife_thresh_inv = knife_pretreatment_inv(knife_original, threshold=args.knife_inv_threshold)
                knife_thresh_inv = cv.cvtColor(knife_thresh_inv, cv.COLOR_GRAY2BGR)
                knife_thresh_inv = knife_thresh_inv[posy:posy + h, posx:posx + w]
                knife_thresh_inv = cv.cvtColor(knife_thresh_inv, cv.COLOR_BGR2GRAY)
                print('knife-thresh-inv:', knife_thresh_inv.shape)

                # 选取和刀具图大小一致的区域为融合区域
                background_real_roi, startX, startY, angle = get_background_real_roi(t, knife, background_original)
                if startX == 0 and startY == 0:
                    # print('第{}图片无法合成'.format(str(j + 1)))
                    continue
                # 制作模板用来筛选区域中的包裹部分(或者说祛除背景)
                bg_thresh_inv = background_pretreatment(background_real_roi, args.bg_inv_threshold)
                print('bg-thresh-inv:', bg_thresh_inv.shape)
                cv.imshow('bg-inv',bg_thresh_inv)

                # 旋转模板
                knife = image_rotation(knife, angle)
                knife_thresh = image_rotation(knife_thresh, angle)
                knife_thresh_inv = image_rotation(knife_thresh_inv, angle)

                # 刀具的HSV图像合成法
                knife_hsv = cv.cvtColor(knife, cv.COLOR_BGR2HSV)
                # 分割通道/dtype:uint8
                k_h, k_s, k_v = cv.split(knife_hsv)
                # 保留刀具图h通道图像中的刀具部分/(knife_thresh/255)的dtype为float64/k_h的dtype是uint8
                k_h_new = k_h * ((knife_thresh / 255).astype(np.uint8))
                k_v_new = k_v * ((knife_thresh / 255).astype(np.uint8))
                k_s_new = k_s * ((knife_thresh / 255).astype(np.uint8))
                # 刀具图要融合的部分
                mm_k = cv.cvtColor(cv.merge([k_h_new, k_s_new, k_v_new]), cv.COLOR_HSV2BGR)
                mm_k = cv.resize(mm_k, (300, 300))
                cv.imshow('kkkkkkkk', mm_k)

                # 背景的HSV图像
                background_hsv = cv.cvtColor(background_real_roi, cv.COLOR_BGR2HSV)
                # 分割通道
                b_h, b_s, b_v = cv.split(background_hsv)
                # 背景图h通道图像中要和刀具融合的区域
                # 去除背景中的影响因素,再制作一个模板只保留背景图中的包裹部分
                b_h_new = b_h * ((bg_thresh_inv / 255).astype(np.uint8))
                b_v_new = b_v * ((bg_thresh_inv / 255).astype(np.uint8))
                b_s_new = b_s * ((bg_thresh_inv / 255).astype(np.uint8))

                bbbbbbb = cv.cvtColor(cv.merge([b_h_new, b_s_new, b_v_new]), cv.COLOR_HSV2BGR)
                bbbbbbb = cv.resize(bbbbbbb, (300, 300))
                cv.imshow('bbbbbbbb1', bbbbbbb)


                b_h_new = b_h_new * ((knife_thresh / 255).astype(np.uint8))
                b_v_new = b_v_new * ((knife_thresh / 255).astype(np.uint8))
                b_s_new = b_s_new * ((knife_thresh / 255).astype(np.uint8))
                # 背景图要融合的部分
                mm_bg = cv.cvtColor(cv.merge([b_h_new, b_s_new, b_v_new]), cv.COLOR_HSV2BGR)
                mm_bg = cv.resize(mm_bg, (300, 300))
                cv.imshow('bbbbbbbb', mm_bg)

                # TODO:根据融合算法创建新的h,s,v通道
                new_mh, new_ms, new_mv, new_m_hsv, new_m_bgr = func1(k_h_new, k_s_new, k_v_new, b_h_new, b_s_new,
                                                                     b_v_new, bg_thresh_inv,n)
                cv.imshow('new_m_bgr', cv.resize(new_m_bgr, (300, 300)))

                # 在原图中各通道融合新的h,s,v
                new_mask_h = ((knife_thresh_inv / 255) * b_h).astype(np.uint8) + new_mh
                new_mask_s = ((knife_thresh_inv / 255) * b_s).astype(np.uint8) + new_ms
                new_mask_v = ((knife_thresh_inv / 255) * b_v).astype(np.uint8) + new_mv
                # 利用新的h,s,v合成HSV图像,再转为BGR图像
                new_m_img = cv.merge([new_mask_h, new_mask_s, new_mask_v])
                new_m_img = cv.cvtColor(new_m_img, cv.COLOR_HSV2BGR)
                new_m_img = new_m_img[1:, 1:]
                cv.imshow('new_m_img', cv.resize(new_m_img, (300, 300)))

                # 截取出的图片有黑边,用裁剪的办法处理掉
                demo = background_c[startY + 1:startY + h, startX + 1:startX + w]
                new_m_bgr = new_m_bgr[1:, 1:]
                new_m_gray = cv.cvtColor(new_m_bgr,cv.COLOR_BGR2GRAY)
                _, thresh = cv.threshold(new_m_gray, 0, 255, cv.THRESH_BINARY_INV)
                cv.imshow('------------------',thresh)
                # 将合成部分的黑边用背景色代替
                for i, height in enumerate(new_m_bgr):
                    for j, width in enumerate(height):
                        if all(new_m_bgr[i][j] == np.array((0, 0, 0))):
                            new_m_bgr[i][j] = demo[i][j]

                # 将新的BGR图像放回原图中
                background_c[startY + 1:startY + h, startX + 1:startX + w] = new_m_bgr
                background_original[startY + 1:startY + h, startX + 1:startX + w] = new_m_bgr
                # print('第{}张图片合成结束...'.format(str(j + 1)))

                # labels文件数据归一化处理
                line = convert(startX, startY, w, h, background_width, background_height)
                cv.rectangle(background_original, (startX - 10, startY - 10), (startX + w + 20, startY + h + 20),
                             (0, 0, 80), thickness=2)
                cv.imshow('res', background_original)
                print('res:', background_original.shape)
                flag = True
                if flag:
                    with open(args.label_save_path + args.c_name + str(NUM_OF_USEFUL_PIC) + '.txt', 'a') as f:
                        f.write(str(n) + ' ' + line + '\n')
                        f.close()
                cv.waitKey(0)

            if flag:
                cv.imwrite(args.img_save_path + args.c_name + str(NUM_OF_USEFUL_PIC) + '.jpg', background_c)
                NUM_OF_USEFUL_PIC = NUM_OF_USEFUL_PIC + 1
                if NUM_OF_USEFUL_PIC == args.total_img:
                    return

        round_over_time = time.time()
        print('第{}轮结束,本轮时间为{}'.format(str(i + 1), round_over_time - round_start_time))

if __name__ == '__main__':
    # 程序开始运行时间
    run_start_time = time.time()
    main()
    run_over_time = time.time()
    print('共用时{}s'.format(run_over_time - run_start_time))
