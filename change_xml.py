import os.path
import xml.dom.minidom

# xml文件存放路径
path = 'E:/dataset/SeeingThroughFog_master_/annotation/train_xml'
# 定义需替换的类名['原始类别名称','新名称']
name = ['LargeVehicle_is_group', 'LargeVehicle']

files = os.listdir(path)  # 返回文件夹中的文件名列表
for xmlFile in files:
    dom = xml.dom.minidom.parse(path + '\\' + xmlFile)
    root = dom.documentElement
    newfilename = root.getElementsByTagName('name')
    for i, t in enumerate(newfilename):
        if t.firstChild.data == name[0]:
            newfilename[i].firstChild.data = name[1]
    with open(os.path.join(path, xmlFile), 'w') as fh:
        dom.writexml(fh)

