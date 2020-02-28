from mxnet import image
from matplotlib import pyplot as plt

img = image.imread('img_01_label.png')
print(img.shape)
img[:, :, 1:3] = 0
print(img)
plt.imsave('test.png', img.asnumpy())
