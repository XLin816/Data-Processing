# # coding： utf-8
# from PIL import Image,ImageFilter
# import os
# from PIL import ImageGrab
# #  源目录
# input_path = 'E:/dataset/val_degamma'
# #  输出目录
# output_path = 'F:/val'
# def imageResize(input_path, output_path):
#     # 获取输入文件夹中的所有文件/夹，并改变工作空间
#     files = os.listdir(input_path)
#     os.chdir(input_path)
#     # 判断输出文件夹是否存在，不存在则创建
#     if (not os.path.exists(output_path)):
#         os.makedirs(output_path)
#     for file in files:
#         # 判断是否为文件，文件夹不操作
#         if (os.path.isfile(file)):
#             img = Image.open(file)
#             #将图片缩放为96*96大小
#             # img = img.resize((640, 640), Image.ANTIALIAS)
#             # 边缘增强
#             img.filter(ImageFilter.EDGE_ENHANCE)
#             # 找到边缘
#             img.filter(ImageFilter.FIND_EDGES)
#             # 浮雕
#             img.filter(ImageFilter.EMBOSS)
#             # 轮廓
#             img.filter(ImageFilter.CONTOUR)
#             # 锐化
#             img.filter(ImageFilter.SHARPEN)
#             # 平滑
#             img.filter(ImageFilter.SMOOTH)
#             # 细节
#             img.filter(ImageFilter.DETAIL)
#             img.save(os.path.join(output_path, "New_" + file))
# imageResize(input_path, output_path)
# import cv2
# import numpy as np
# picture = cv2.imread("000046.png")
# gray = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
# edge = cv2.Canny(gray,100,250,3)
# (thresh, blackEdges) = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY_INV)
# # result = blackEdges + gray   # I tryed this and ,i didn't get what i want.
# result = picture.copy()
# result[blackEdges==0] = (0,0,0)
# cv2.imshow("Result",result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from scipy import misc, ndimage
import numpy as np
from matplotlib import pyplot as plt
import cv2

# im = cv2.imread('car-Ice.jpg') / 255 # scale pixel values in [0,1] for each channel
#
# # First a 1-D  Gaussian
# t = np.linspace(-10, 10, 30)
# bump = np.exp(-0.1*t**2)
# bump /= np.trapz(bump) # normalize the integral to 1
#
# # make a 2-D kernel out of it
# kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
#
# im_blur = ndimage.convolve(im, kernel.reshape(30,30,1))
#
# im_sharp = np.clip(2*im - im_blur, 0, 1)
#
# fig, ax = plt.subplots(nrows=2, figsize=(10, 20))
#
# ax[0].imshow(im)
# ax[0].set_title('Original Image', size=20)
#
# ax[1].imshow(im_sharp)
# ax[1].set_title('Sharpened Image', size=20)
#
# plt.show()

from scipy import misc, signal
import numpy as np
import os, random

myPath = r'C:\Users\Holmes\Desktop\imagenhance\test'
#输出目录]
outPath = r'C:\Users\Holmes\Desktop\imagenhance\test'

def Sharpen(filesource, outputfile, name, imgtype):
    imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
    # 打开图片
    name_i = os.path.join(filesource, name)
    im = cv2.imread(name_i)/255
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    im_sharpened = np.ones(im.shape)
    for i in range(3):
        im_sharpened[...,i] = np.clip(signal.convolve2d(im[...,i], sharpen_kernel, mode='same', boundary="symm"),0,1)
    defog_name = os.path.join(outputfile, name)
    cv2.imwrite(defog_name, im_sharpened * 255)


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
        # print(postfix, i)
        # name2 = os.path.join(myPath, i)
        # print(name2)
        # if postfix == '.jpg' or postfix == '.png':
        Sharpen(myPath, outPath, i, postfix)
# im = cv2.imread('car-Ice.jpg')/255. # scale pixel values in [0,1] for each channel
#
# print(np.max(im))
# # 1.0
# print(im.shape)
# # (220, 220, 3)
#
# sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# im_sharpened = np.ones(im.shape)
# for i in range(3):
#     im_sharpened[...,i] = np.clip(signal.convolve2d(im[...,i], sharpen_kernel, mode='same', boundary="symm"),0,1)

# fig, ax = plt.subplots(nrows=2, figsize=(10, 20))
# plt.figure()
# plt.imshow(im)
# # ax[0].set_title('Original Image', size=20)
# plt.figure()
# plt.imshow(im_sharpened)
# cv2.imwrite('Shapened-Image.jpg', im_sharpened * 255)
# # ax[1].set_title('Sharpened Image', size=20)
# plt.show()
if __name__ == '__main__':
    run()