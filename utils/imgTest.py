from mxnet import image, nd
from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np

# img = nd.random.uniform(0, 255, shape=(128, 128, 3))
# img_path = os.path.join('.', 'test_01.jpeg')
# img = img.astype('int32').asnumpy()
img = np.random.random_integers(0, 255, size=(128, 128, 3))
print(img.shape)
img = np.asarray(img, dtype='uint8')
plt.imsave('test_01.tif', img)
# plt.imsave(img_path, img)
# srcImg = image.imread('z0.tif')
# print(srcImg)
