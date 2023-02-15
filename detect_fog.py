from PIL import Image
import glob, time
import os
import shutil
# data_path = 'C:/Users/Holmes/Desktop/Reside/RTTS/JPEGImages'
# move_path = 'C:/Users/Holmes/Desktop/Reside/RTTS/fog'
def slow_horizontal_variance(im):
    '''Return average variance of horizontal lines of a grayscale image'''
    width, height = im.size
    if not width or not height: return 0
    vars = []
    pix = im.load()
    for y in range(height):
        row = [pix[x,y] for x in range(width)]
        mean = sum(row)/width
        variance = sum([(x-mean)**2 for x in row])/width
        vars.append(variance)
    return sum(vars)/height

def run():
    pathDir = os.listdir(data_path)
    for i in pathDir:
        foggy_name = os.path.join(data_path, i)
        # print(foggy_name)
        for fn in glob.glob(foggy_name):
            im = Image.open(fn).convert('L')
            var = slow_horizontal_variance(im)
            # fog = var < 1000    # FOG THRESHOLD
            fog = var
            if fog < 712:
                print('%5.0f - %5s - %s' % (var, fog and 'FOGGY' or 'SHARP', fn))
                shutil.move(foggy_name, move_path + '/' + i)

if __name__ == '__main__':
    run()
# time_start = time.time()
# for fn in glob.glob(r'000041.jpg'):
#     im = Image.open(fn).convert('L')
#     var = slow_horizontal_variance(im)
#     fog = var < 1000    # FOG THRESHOLD
#     time_end = time.time()
#     time_sum = time_end - time_start
#     print(time_sum)
#     print('%5.0f - %5s - %s' % (var, fog and 'FOGGY' or 'SHARP', fn))
