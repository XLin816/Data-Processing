import shutil
import os


def objFileName():
    local_file_name_list = "F:/fogdata_voc/test - normal.txt"
    obj_name_list = []
    for i in open(local_file_name_list, 'r'):
        obj_name_list.append(i.replace('\n', ''))
    return obj_name_list


def copy_img():
    local_img_name = r'D:\fogdata_voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
    # 指定要复制的图片路径
    path = r'C:\Users\Holmes\Desktop\test'
    # 指定存放图片的目录
    for i in objFileName():
        new_obj_name = i
        # dir, file = os.path.split(new_obj_name)
        shutil.copy(local_img_name+'/'+new_obj_name, path + '/' + new_obj_name)
        # shutil.move(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)


if __name__ == '__main__':
    copy_img()
