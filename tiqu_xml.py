'''
the function of the code
批量提取指定类别的xml和对应图片
author@bjtu_huangyuxiang
'''
from __future__ import division
import os
import xml.dom.minidom
import shutil

classes = ['person', 'car', 'bus', 'bicycle',  'motorbike']
dirnames = ['xml_2007', 'images_2007']


def extraction_box(ImgPath, AnnoPath, OutPath):
    # 判断输出路径是否存在，不存在的话就新建
    output_subdir = os.path.join(OutPath)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    outpath1 = os.path.join(output_subdir, dirnames[0]) + '/'
    if not os.path.exists(outpath1):
        os.makedirs(outpath1)
    outpath2 = os.path.join(output_subdir, dirnames[1]) + '/'
    if not os.path.exists(outpath2):
        os.makedirs(outpath2)

    i = 0
    imagelist = os.listdir(ImgPath)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)  # 分割文件名和后缀
        imgfile = ImgPath + image_pre + '.jpg'
        if not os.path.exists(AnnoPath + image_pre + '.xml'):
            continue
        xmlfile = AnnoPath + image_pre + '.xml'
        DomTree = xml.dom.minidom.parse(xmlfile)
        root = DomTree.documentElement  # 获得xml文档对象
        objectlist = root.getElementsByTagName('object')  # 获得xml文档目标名称
        for objects in objectlist:
            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data
            # print(objectname)
            if objectname == classes[0]:
                shutil.copy(xmlfile, outpath1)
                shutil.copy(imgfile, outpath2)
                print("提取%s,%s" % (imgfile, xmlfile))
                i = i + 1
    print("%s files have been selected!" % i)


if __name__ == "__main__":
    ImgPath = r'D:/fogdata_voc/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    AnnoPath = r'D:/fogdata_voc/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations/'
    OutPath = r'F:\fogdata_voc\train'
    extraction_box(ImgPath, AnnoPath, OutPath)
