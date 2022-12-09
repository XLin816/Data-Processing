import os
#paths=['coco2014','Stanford','vehicleplate']
paths=['F:/fogdata_voc/images/trainval/']
f=open('F:/fogdata_voc/images/trainval.txt', 'w')
for path in paths:
    # p=os.path.abspath(path)+'/'
    p = os.path.abspath(path)
    # p = './dataset' + '/images/train'
    # p1 = os.path.abspath(path)
    filenames=os.listdir(p)
    for filename in filenames:
        im_path=p+'/'+filename
        print(im_path)
        f.write(im_path+'\n')
f.close()