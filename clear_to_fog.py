import os, random

import matplotlib.pyplot as plt
from PIL import Image
import cv2, math
import numpy as np
from numba import jit
# 源目录
myPath = r'F:\fogdata_voc\images\test'
#输出目录]
outPath = r'F:\fogdata_voc\images\test'

def processImage(filesource, destsource, name, imgtype):
    '''
    filesource是存放待雾化图片的目录
    destsource是存放物化后图片的目录
    name是文件名
    imgtype是文件类型
    '''
    imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
    # 打开图片
    name_i = os.path.join(filesource, name)
    img = cv2.imread(name_i)
    for i in range(10):
        @jit()
        def AddHaz_loop(img_f, center, size, beta, A):
            (row, col, chs) = img_f.shape

            for j in range(row):
                for l in range(col):
                    d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                    td = math.exp(-beta * d)
                    img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
            return img_f

        img_f = img / 255.0
        (row, col, chs) = img.shape

        A = 0.5  # 亮度
        # beta = 0.09  # 雾的浓度
        beta = 0.01 * i + 0.04  # 雾的浓度
        size = math.sqrt(max(row, col))  # 雾化尺寸
        center = (row // 2, col // 2)  # 雾化中心
        foggy_image = AddHaz_loop(img_f, center, size, beta, A)
        img_f = np.clip(foggy_image * 255, 0, 255)
        img_f = img_f.astype(np.uint8)
        # for j in range(row):
        #     for l in range(col):
        #         d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
        #         td = math.exp(-beta * d)
        #         img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
        # cv2.imwrite(destsource + name, img_f * 255)
        foggy_name = os.path.join(destsource, name)
        cv2.imwrite(foggy_name, img_f)


def run():
    # 切换到源目录，遍历目录下所有图片
    #os.chdir(myPath)
    pathDir = os.listdir(myPath) #提取文件内文件名
    # print(int(len(pathDir) * 1 / 3)) #获取文件内图片总数
    # sample = random.sample(pathDir, 1) #随机抽取1/3图片雾化
    sample = random.sample(pathDir, int(len(pathDir)))
    # sample = random.sample(pathDir, int(len(pathDir)))
    # for i in os.listdir(os.getcwd()):
    for i in sample:
        # 检查后缀
        postfix = os.path.splitext(i)[1]
        print(postfix, i)
        # name2 = os.path.join(myPath, i)
        # print(name2)
        # if postfix == '.jpg' or postfix == '.png':
        processImage(myPath, outPath, i, postfix)


if __name__ == '__main__':
    run()


# import os
#
# filePath = './data/images/test'
# filenames = os.listdir(filePath)
#
# outputPath = './data/images/clear_to_fog'
#
# #     img_path = 'test.png'
# #     img = cv2.imread(img_path)
# #     img_f = img / 255.0
# #     (row, col, chs) = img.shape
# for filename in filenames:
#     print(filename)
#     # img = cv2.imread(filePath + filename)
#     img = cv2.imread(filename)
#     img_f = img / 255.0
#     (row, col, chs) = img.shape
#
#     A = 0.5  # 亮度
#     beta = 0.08  # 雾的浓度
#     size = math.sqrt(max(row, col))  # 雾化尺寸
#     center = (row // 2, col // 2)  # 雾化中心
#     for j in range(row):
#         for l in range(col):
#             d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
#             td = math.exp(-beta * d)
#             img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
#
#     cv2.imwrite(outputPath + filename, img_f * 255)