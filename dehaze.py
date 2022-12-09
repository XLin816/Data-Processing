# # -*- coding: utf-8 -*-
# """
# Created on Sat Sep 11 00:16:07 2021
# @author: xiuzhang
#
# 参考资料：
# https://blog.csdn.net/leviopku/article/details/83898619
# """
#
import sys
# import cv2, os, random
# import math,time
# import numpy as np
#
# myPath = r'C:/Users/Holmes/Desktop/Reside/RTTS/fog'
# #输出目录]
# outPath = r'C:/Users/Holmes/Desktop/Reside/RTTS/dehaze_fog'
#
# def DarkChannel(im, sz):
#     b, g, r = cv2.split(im)
#     dc = cv2.min(cv2.min(r, g), b)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
#     dark = cv2.erode(dc, kernel)
#     return dark
#
#
# def AtmLight(im, dark):
#     [h, w] = im.shape[:2]
#     imsz = h * w
#     numpx = int(max(math.floor(imsz / 1000), 1))
#     darkvec = dark.reshape(imsz, 1)
#     imvec = im.reshape(imsz, 3)
#
#     indices = darkvec.argsort()
#     indices = indices[imsz - numpx::]
#
#     atmsum = np.zeros([1, 3])
#     for ind in range(1, numpx):
#         atmsum = atmsum + imvec[indices[ind]]
#
#     A = atmsum / numpx
#     return A
#
#
# def TransmissionEstimate(im, A, sz):
#     omega = 0.95
#     im3 = np.empty(im.shape, im.dtype)
#
#     for ind in range(0, 3):
#         im3[:, :, ind] = im[:, :, ind] / A[0, ind]
#
#     transmission = 1 - omega * DarkChannel(im3, sz)
#     return transmission
# #
# #
# def Guidedfilter(im, p, r, eps):
#     mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
#     mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
#     mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
#     cov_Ip = mean_Ip - mean_I * mean_p
#
#     mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
#     var_I = mean_II - mean_I * mean_I
#
#     a = cov_Ip / (var_I + eps)
#     b = mean_p - a * mean_I
#
#     mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
#     mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
#
#     q = mean_a * im + mean_b
#     return q
#
#
# def TransmissionRefine(im, et):
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     gray = np.float64(gray) / 255
#     r = 60
#     eps = 0.0001
#     t = Guidedfilter(gray, et, r, eps)
#
#     return t
#
#
# def Recover(im, t, A, tx=0.1):
#     res = np.empty(im.shape, im.dtype)
#     t = cv2.max(t, tx)
#
#     for ind in range(0, 3):
#         res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
#
#     return res
#
# def Dehaze(filesource, outputfile, name, imgtype):
#     imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
#     # 打开图片
#     name_i = os.path.join(filesource, name)
#     print(name_i)
#     img_1 = cv2.imread(name_i)
#     I = img_1.astype('float64') / 255
#
#     dark = DarkChannel(I, 15)
#     A = AtmLight(I, dark)
#     te = TransmissionEstimate(I, A, 15)
#     t = TransmissionRefine(img_1, te)
#     J = Recover(I, t, A, 0.1)
#     #
#     arr = np.hstack((I, J))
#     defog_name = os.path.join(outputfile, name)
#     cv2.imwrite(defog_name, J * 255)
#
#     # img = cv2.imread(defog_name, 1)
#     #
#     # imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     # channelsYUV = cv2.split(imgYUV)
#     # t = channelsYUV[0]
#     #
#     # # 限制对比度的自适应阈值均衡化
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     # p = clahe.apply(t)
#     #
#     # channels = cv2.merge([p, channelsYUV[1], channelsYUV[2]])
#     # result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
#     # cv2.imwrite(defog_name, result)
#
#
# def run():
#     # 切换到源目录，遍历目录下所有图片
#     #os.chdir(myPath)
#     pathDir = os.listdir(myPath) #提取文件内文件名
#     # print(int(len(pathDir) * 1 / 3)) #获取文件内图片总数
#     # sample = random.sample(pathDir, 1) #随机抽取1/3图片雾化
#     sample = random.sample(pathDir, int(len(pathDir)))
#     # sample = random.sample(pathDir, int(len(pathDir)))
#     # for i in os.listdir(os.getcwd()):
#     for i in sample:
#         # 检查后缀
#         postfix = os.path.splitext(i)[1]
#         # print(postfix, i)
#         # name2 = os.path.join(myPath, i)
#         # print(name2)
#         # if postfix == '.jpg' or postfix == '.png':
#         Dehaze(myPath, outPath, i, postfix)
# #
# # # 主函数
# if __name__ == '__main__':
# # #     # img = cv2.imread('000739.jpg')
# # #     # res = zmIceColor(img / 255.0) * 255
# # #     # cv2.imwrite('car-Ice.jpg', res)
#     run()

# if __name__ == '__main__':
#     time_start = time.time()
#
#     fn = '000046.png'
#     src = cv2.imread(fn)
#     I = src.astype('float64') / 255
#
#     dark = DarkChannel(I, 15)
#     A = AtmLight(I, dark)
#     te = TransmissionEstimate(I, A, 15)
#     t = TransmissionRefine(src, te)
#     J = Recover(I, t, A, 0.1)
# #
#     arr = np.hstack((I, J))
# #
#     # cv2.imshow("contrast", arr)
#     cv2.imwrite("car-02-dehaze.png", J * 255)
#     time_end = time.time()
#     time_sum = time_end - time_start
#     print(time_sum)
#     cv2.imwrite("car-02-contrast.png", arr * 255)
#     img = cv2.imread("car-02-dehaze.png", 1)
#
#     imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     channelsYUV = cv2.split(imgYUV)
#     t = channelsYUV[0]
#
#     # 限制对比度的自适应阈值均衡化
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     p = clahe.apply(t)
#
#     channels = cv2.merge([p, channelsYUV[1], channelsYUV[2]])
#     result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
#     cv2.imshow("dst", result)
#     cv2.imwrite("E:/dataset/kitti/result.jpg", result)  # 图像保存位置
    # cv2.waitKey();

# -*- coding: utf-8 -*-
# By:Eastmount CSDN 2021-03-12
# 惨zmshy2128老师文章并修改成Python3代码
import cv2
import numpy as np
import math
import time
import os, random
import matplotlib.pyplot as plt
from skimage import exposure
from PIL import Image

# myPath = r'C:/Users/Holmes/Desktop/Reside/RTTS/fog'
# #输出目录]
# outPath = r'C:/Users/Holmes/Desktop/Reside/RTTS/dehaze_fog'

# 线性拉伸处理
# 去掉最大最小0.5%的像素值 线性拉伸至[0,1]
def stretchImage(data, s=0.005, bins=2000):
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


# 根据半径计算权重参数矩阵
g_para = {}


def getPara(radius=5):
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


# 常规的ACE实现
def zmIce(I, ratio=4, radius=300):
    para = getPara(radius)
    height, width = I.shape
    zh = []
    zw = []
    n = 0
    while n < radius:
        zh.append(0)
        zw.append(0)
        n += 1
    for n in range(height):
        zh.append(n)
    for n in range(width):
        zw.append(n)
    n = 0
    while n < radius:
        zh.append(height - 1)
        zw.append(width - 1)
        n += 1
    # print(zh)
    # print(zw)

    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


# 单通道ACE快速增强实现
def zmIceFast(I, ratio, radius):
    print(I)
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, (int((width + 1) / 2), int((height + 1) / 2)))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


# rgb三通道分别增强 ratio是对比度增强因子 radius是卷积模板半径
def zmIceColor(I, ratio=4, radius=3):
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res
#
# def Dehaze(filesource, outputfile, name, imgtype):
#     imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
#     # 打开图片
#     name_i = os.path.join(filesource, name)
#     img = cv2.imread(name_i)
#     res = zmIceColor(img / 255.0) * 255
#     defog_name = os.path.join(outputfile, name)
#     cv2.imwrite(defog_name, res)
#
# def run():
#     # 切换到源目录，遍历目录下所有图片
#     #os.chdir(myPath)
#     pathDir = os.listdir(myPath) #提取文件内文件名
#     # print(int(len(pathDir) * 1 / 3)) #获取文件内图片总数
#     # sample = random.sample(pathDir, 1) #随机抽取1/3图片雾化
#     sample = random.sample(pathDir, int(len(pathDir)))
#     # sample = random.sample(pathDir, int(len(pathDir)))
#     # for i in os.listdir(os.getcwd()):
#     for i in sample:
#         # 检查后缀
#         postfix = os.path.splitext(i)[1]
#         print(postfix, i)
#         # name2 = os.path.join(myPath, i)
#         # print(name2)
#         # if postfix == '.jpg' or postfix == '.png':
#         Dehaze(myPath, outPath, i, postfix)

# 主函数
if __name__ == '__main__':
    # time_start = time.clock()
    img = cv2.imread('3s.tif', 1)
    res = zmIceColor(img / 255.0) * 255
    # time_end = time.clock()
    # time_sum = time_end - time_start
    # print(time_sum)
    cv2.imwrite('car-Ice.jpg', res)

    # run()
#     # 自适应直方图均衡化(AHE)
#
#     # file_to_open = 'car-02-dehaze.png'
#     # img2 = cv2.imread(file_to_open)
#     # # plt.figure()
#     # # plt.imshow(img2)
#     # # plt.show()
#     #
#     # img = exposure.equalize_adapthist(img2)
#     # im = Image.fromarray(np.uint8(img * 255))
#     # im.save('result2.jpg')
#     # plt.figure()
#     # plt.imshow(img)
#     # plt.show()
#
#     # img2 = cv2.imread('car-Ice.jpg')
#     # imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     # channelsYUV = cv2.split(imgYUV)
#     # t = channelsYUV[0]
#     #
#     # # 限制对比度的自适应阈值均衡化
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     # p = clahe.apply(t)
#     #
#     # channels = cv2.merge([p, channelsYUV[1], channelsYUV[2]])
#     # result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
#     #
#     # cv2.imshow("dst", result)
#     # cv2.imwrite("result.jpg", result)  # 图像保存位置