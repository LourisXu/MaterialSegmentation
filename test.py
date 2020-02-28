from unet import UNet
import os
from mxnet import nd, image

# data
data_dir = os.path.join('.', 'test_data')
model_dir = os.path.join('.', 'checkpoints', 'class_02')
save_txt = os.path.join('.', 'test.txt')
imgs_list = os.listdir(data_dir)
# model
net = UNet()
net.load_parameters(os.path.join(model_dir, 'epoch_499_model_parameters.params'))

rows = 2*512*512

len = len(imgs_list)
feature_map = None


def processImage(img):
    rgb_mean = nd.array([0.485, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])
    return (img.astype('float32') / 255 - rgb_mean) / rgb_std


with open(save_txt, 'w') as f:
    for i, img_name in enumerate(imgs_list):
        img = image.imread(os.path.join(data_dir, img_name))
        img = processImage(img)
        img = img.transpose((2, 0, 1)).expand_dims(axis=0)
        img = net(img).flatten()
        # print(img.shape)
        if feature_map is None:
            feature_map = img
        else:
            feature_map = nd.concat(feature_map, img, dim=0)
    # print(feature_map.shape)
    for row in range(rows):
        s = ''
        for col in range(len):
            if col != 0:
                s += '\t'
            s += str(feature_map[col][row].asscalar())
            # print(s)
        print('%s' % s, file=f)
