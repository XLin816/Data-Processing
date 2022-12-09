# f = open('./splits/test_clear_day - 2.txt')
# lines = f.readlines() #整行读取
# f.close()
# for line in lines:
#     rs = line.rstrip('\n') #去除原来每行后面的换行符，但有可能是\r或\r\n
#     newname=rs.replace(rs,rs+'.png')
#     newfile=open('./splits/test_clear_day - 3.txt','a')
#     newfile.write(newname+'\n')
#     newfile.close()
import os

#D:/foggy_data/labels/train
myPath = 'C:/Users/Holmes/Desktop/Reside/RTTS/labels/labelscl' #源目录
savePath = 'C:/Users/Holmes/Desktop/Reside/RTTS/labels/test' #输出目录

# pathDir = os.listdir(myPath) #提取文件内文件名
# f = open('./splits/test.txt')
# lines = f.readlines() #整行读取
# f.close()

# 类别标签转换为数字标签
def str2num(s):
    # digits = {'Person': 0,  'Car': 1, 'Van': 2, 'Cyclist': 3, 'Tram': 4, 'Misc': 5, 'Truck':6,
    #     'PassengerCar': 7, 'LargeVehicle': 8, 'RidableVehicle': 9,'Person_sitting':10, 'Adult':11, 'Children':12, 'Limousine':13, 'SUV':14, 'Sport Car':15, 'Cabriolet': 16, 'Coupe':17, 'Hatchback':18, 'Caravan':19,
    #     'Trailor':20, 'Bus':21, 'Bicycle':22, 'Tricycle':23, 'Motocycle':24,'Scooter':25,'Wheel Chair':26, 'Buggy':27, 'Quad':28, 'Vehicle':29, 'DontCare':30,
    #     'Obstacle':31,'Pedestrian_is_group':32, 'PassengerCar_is_group':33, 'RidableVehicle_is_group':34, 'Vehicle_is_group':35, 'train':36, 'LargeVehicle_is_group':37}
    # digits = {'Car': 0, 'DontCare': 8, 'Pedestrian': 1, 'Truck': 2, 'Van': 3, 'Misc': 4, 'Cyclist': 5, 'Tram': 6, 'Person_sitting': 7}
    digits = {'car': 0, 'person': 1, 'bicycle': 2, 'bus': 3, 'motorbike': 4}
    return digits[s]

def labelchage(firlsource, outputfile, name):
    pathname = os.path.join(firlsource, name)
    f = open(pathname)
    lines = f.readlines()  # 整行读取
    f.close()
    for line in lines:
        rs = line.rstrip('\n')  # 去除原来每行后面的换行符，但有可能是\r或\r\n
        str_label = rs.split(' ')[0]  # 读取类别
        # print(str_label)
        # id_list = map(str2num, str)
        newname = rs.replace(str_label, str(str2num(str_label)))  # 数字标签替换
        # print(newname)
        newfilename = os.path.join(outputfile, name)
        newfile = open(newfilename, 'a')
        newfile.write(newname + '\n')
        newfile.close()

def run():
    pathDir = os.listdir(myPath)  # 提取文件内文件名
    for i in pathDir:
        # f = open(i)
        # lines = f.readlines()  # 整行读取
        # f.close()
        # for line in lines :
        #     rs = line.rstrip('\n') #去除原来每行后面的换行符，但有可能是\r或\r\n
        #     str_label = rs.split(' ')[0] #读取类别
        #     # print(str_label)
        #     # id_list = map(str2num, str)
        #     newname = rs.replace(str_label, str(str2num(str_label))) #数字标签替换
        #     # print(newname)
        #     newfile = open('./splits/test_1.txt', 'a')
        #     newfile.write(newname+'\n')
        #     newfile.close()
        labelchage(myPath, savePath, i)

if __name__ == '__main__':
    run()

