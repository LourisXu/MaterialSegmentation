from mxnet import image
from matplotlib import pyplot as plt
import os

data_dir = os.path.join('..', 'data', 'W800_147')
src_dir = os.path.join(data_dir, 'label')
dst_dir = os.path.join(data_dir, 'label_png')
img_list = os.listdir(src_dir)
for i in range(len(img_list)):
    srcImg = image.imread(os.path.join(src_dir, img_list[i]))
    srcImg = srcImg.asnumpy()
    mask = (srcImg > 0)
    srcImg[mask] = 255
    plt.imsave(os.path.join(dst_dir, '%s_label.png' % img_list[i][:-4]), srcImg)
