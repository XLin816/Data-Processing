#  批量移除xml标注中的某一个类别标签
import xml.etree.cElementTree as ET
import os

# xml文件路径
xml_path = r'F:\fogdata_voc\annotation\test'
new_path = r'F:\fogdata_voc\annotation\test'
xml_files = os.listdir(new_path)
# 需要删除的类别名称
CLASSES = ['aeroplane','bird', 'boat', 'bottle',  'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

for axml in xml_files:
    path_xml = os.path.join(new_path, axml)
    tree = ET.parse(path_xml)
    root = tree.getroot()
    for child in root.findall('object'):
        name = child.find('name').text
        if name in CLASSES:
            root.remove(child)
    tree.write(os.path.join(new_path, axml))

