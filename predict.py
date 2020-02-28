from mxnet.gluon import data as gdata
from mxnet import nd, image
from load import load_data, SegDataset, COLORMAP
from unet import UNet
import mxnet as mx
import os
from matplotlib import pyplot as plt


def predict(img):
    rgb_mean = nd.array([0.485, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])
    X = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    X = X.transpose((2, 0, 1)).expand_dims(axis=0)
    # print(X.shape)
    pred = nd.argmax(net(X.as_in_context(ctx[0])), axis=1)
    # print('pred shape(pre):', pred.shape)
    return pred.reshape((pred.shape[1], pred.shape[2]))


def label2image(pred):
    # print('pred shape:', pred.shape)
    colormap = nd.array(COLORMAP, ctx=ctx[0], dtype='uint8')
    # print('colormap shape:', colormap.shape)
    X = pred.astype('int32')
    return colormap[X, :]


if __name__ == '__main__':
    # features, labels = load_data()
    # test_features, test_labels = features[-100:], labels[-100:]
    # data_dir = os.path.join('.', 'data', 'class_01', 'image')
    # features_names = [name[:-4] for name in os.listdir(data_dir)]
    # test_names = features_names[-100:]
    class_dir = 'class_05'
    src_dir = 'test'
    # dst_dir = 'Tile_r1-c2_L1'
    data_dir = os.path.join('.', 'data', 'new_512_2')
    image_dir = os.path.join(data_dir, 'image')
    test_files = os.listdir(image_dir)
    test_names = [name[:-4] for name in test_files]
    test_features = [None] * len(test_files)
    for i, fname in enumerate(test_files):
        # print("Loading %s" % fname)
        test_features[i] = image.imread(os.path.join(image_dir, fname))

    print('All %d images have been loaded!' % len(test_features))
    ctx = [mx.cpu()]
    net = UNet(num_class=3)
    file_name = os.path.join('.', 'checkpoints', class_dir, 'epoch_499_model_parameters.params')
    net.load_parameters(file_name)
    net.collect_params().reset_ctx(ctx)
    save_dir = os.path.join(data_dir, 'predict_label')

    n = len(test_features)
    for i in range(n):
        pred = label2image(predict(test_features[i]))
        save_name = test_names[i] + '_predict_label.tif'
        plt.imsave(os.path.join(save_dir, save_name), pred.asnumpy())
        # print('%s has been saved!' % save_name)
        if (i + 1) % 100 == 0:
            print('%d labels have been saved!' % (i + 1))
