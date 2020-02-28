from mxnet import image, nd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc

name = '0850_1050'
dir_name = 'W800_147'
img_path = os.path.join('..', 'data', dir_name, 'image', '%s.tif' % name)
srcImg = image.imread(img_path)
print(type(srcImg))
print(srcImg.shape)
newImg = nd.zeros(shape=srcImg.shape)
# print(newImg)
height, width, _ = srcImg.shape
print(height, width)

SIZE = 128
rows = height // SIZE
cols = width // SIZE
print(rows, cols)
data_dir = os.path.join('..', 'data', dir_name, 'z4')
dst_dir = os.path.join('..', 'data', dir_name, 'predict_label')
img_list = os.listdir(data_dir)
print(img_list)
for i in range(len(img_list)):
    img_name = img_list[i]
    tmpImg = image.imread(os.path.join(data_dir, img_name))
    print(tmpImg.shape)
    img_name = img_name[:-18]
    print(img_name)
    rPos = img_name.rfind('_', 0, len(img_name))
    lPos = img_name.rfind('_', 0, rPos - 1)
    # print(lPos, rPos)
    row = int(img_name[lPos + 1:rPos])
    col = int(img_name[rPos + 1:])
    print(row, col)
    if row < rows and col < cols:
        newImg[row * SIZE:(row + 1) * SIZE, col * SIZE:(col + 1) * SIZE, :] = tmpImg[:, :, :]
    elif row < rows and col == cols:
        newImg[row * SIZE:(row + 1) * SIZE, -SIZE:, :] = tmpImg[:, :, :]
    elif row == rows and col < cols:
        newImg[-SIZE:, col * SIZE:(col + 1) * SIZE, :] = tmpImg[:, :, :]
    else:
        newImg[-SIZE:, -SIZE:, :] = tmpImg[:, :, :]
newImg /= 255
plt.imsave(os.path.join(dst_dir, 'W800_147_%s_label.png' % name), newImg.asnumpy())
