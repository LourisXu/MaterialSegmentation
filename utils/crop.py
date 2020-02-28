# Crop the images to the right size
# because the origin size of each image is too large
# to train in our convolutional neural network,
# which requires too much GPU memory.

# If you want to crop your own images,
# just modify the relevant parameters

import os
import re
from mxnet import image
from matplotlib import pyplot as plt
import numpy as np
import math
import cv2 as cv

root_dir = os.path.join('..', 'data')
srcDir_name = 'TCP'
class_name = 'class_06'
dstDir_name = 'total'
bImage = False

if bImage:
    dst_dir = 'image'
else:
    dst_dir = 'label'

image_dir = os.path.join(root_dir, srcDir_name, dst_dir)
save_dir = os.path.join(root_dir, class_name, dstDir_name, dst_dir)


SIZE = 512 
img_list = os.listdir(image_dir)
print('All images:', img_list)
for i in range(len(img_list)):
    img_name = img_list[i]
    img_path = os.path.join(image_dir, img_name)
    # img = image.imread(img_path)
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    print('Img shape:', img.shape)
    print(type(img))
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    rows = math.ceil(img.shape[0] / float(SIZE))
    cols = math.ceil(img.shape[1] / float(SIZE))
    print('The %d-th img(%s)' % (i, img_name))
    print('Rows, Cols:', rows, cols)
    if not bImage:
        mask = (img > 0)
        img[mask] = 255
        # img[:, :, 0:2] = 0
    new_img = None
    for row in range(rows):
        for col in range(cols):
            if row == rows - 1 and col != cols - 1:
                new_img = img[img.shape[0] - SIZE:, col * SIZE: (col + 1) * SIZE, :]
            if row == rows - 1 and col == cols - 1:
                new_img = img[img.shape[0] - SIZE:, img.shape[1] - SIZE:, :]
            if row != rows - 1 and col != cols - 1:
                new_img = img[row * SIZE: (row + 1) * SIZE, col * SIZE:(col + 1) * SIZE, :]
            if row != rows - 1 and col == cols - 1:
                new_img = img[row * SIZE: (row + 1) * SIZE, img.shape[1] - SIZE:, :]
            new_img_name = srcDir_name + '_' + img_name[:-4] + '_' + str(row) + '_' + str(col) + '.tif'
            save_path = os.path.join(save_dir, new_img_name)
            cv.imwrite(save_path, new_img)
            print('The (%d, %d) patch (%s) has been saved. ' % (row, col, new_img_name))

