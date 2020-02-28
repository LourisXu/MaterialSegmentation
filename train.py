from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn, utils as gutils
import numpy as np
import mxnet as mx
import sys
import os
import tarfile
import time
from load import load_data, SegDataset, colormap2label
from unet import UNet


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    # 类似对角矩阵
    return nd.array(weight)


def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return gutils.split_and_load(features, ctx), gutils.split_and_load(labels, ctx), features.shape[0]


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for x, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(x).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    with open('class_06_log.txt', 'w') as f:
        print('training on', ctx)
        print('training on', ctx, file=f)
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
            for i, batch in enumerate(train_iter):
                xs, ys, batch_size = _get_batch(batch, ctx)
                ls = []
                with autograd.record():
                    y_hats = [net(x) for x in xs]
                    ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
                for l in ls:
                    l.backward()
                trainer.step(batch_size)
                train_l_sum += sum([l.sum().asscalar() for l in ls])
                n += sum([l.size for l in ls])

                train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
                m += sum([y.size for y in ys])
            if epoch != 0 and (epoch + 1) % 100 == 0:
                save_dir = os.path.join('.', 'checkpoints')
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                file_name = os.path.join(save_dir, 'epoch_%d_model_parameters.params' % epoch)
                net.save_parameters(file_name)
            test_acc = evaluate_accuracy(test_iter, net, ctx)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start), file=f)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # parameters

    batch_size = 4
    num_workers = 0 if sys.platform.startswith('win') else 4
    # load data
    train_features, train_labels, test_features, test_labels = load_data()
    train_iter = gdata.DataLoader(SegDataset(train_features, train_labels, colormap2label),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(SegDataset(test_features, test_labels, colormap2label),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    # model
    # model_dir = os.path.join('.', 'model', 'resnet18_v2-a81db45f')
    # pretrained_net = model_zoo.vision.resnet18_v2(root=model_dir, pretrained=True)
    #
    # # FCN (Full Convolutional Network)
    # net = nn.Sequential()
    # for layer in pretrained_net.features[:-2]:
    #     net.add(layer)
    # num_classes = 2
    #
    # net.add(nn.Conv2D(num_classes, kernel_size=1),
    #         nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))
    # net[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 64)))
    # net[-2].initialize(init=init.Xavier())
    net = UNet(num_class=4)
    ctx = [mx.gpu(0)]
    net.initialize(init=init.Xavier(), ctx=ctx)

    # train and test
    loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
    # net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001, 'wd': 1e-3})
    train(train_iter=train_iter, test_iter=test_iter, net=net, loss=loss, trainer=trainer, ctx=ctx, num_epochs=500)
