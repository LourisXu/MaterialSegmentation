from mxnet import image, nd
from mxnet.gluon import data as gdata
import os

COLORMAP = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]]
CLASSES = ['background', 'material_01', 'material_02', 'material_03']
colormap2label = nd.zeros(256 ** 3)
for i, colormap in enumerate(COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


def load_file(img_class, image_dir, label_dir):
    print('Loading the %s images and labels' % img_class)
    image_list = os.listdir(image_dir)
    features, labels = [None] * len(image_list), [None] * len(image_list)
    for i, file in enumerate(image_list):
        features[i] = image.imread(os.path.join(image_dir, file))
        labels[i] = image.imread(os.path.join(label_dir, file))
        print('%s has been loaded!' % file)
    print('All %d %s images and labels have been loaded!' % (len(image_list), img_class))
    return features, labels, len(features)


def load_data():
    root_dir = os.path.join('.', 'data')
    class_dir = os.path.join('class_06')

    train_image_dir = os.path.join(root_dir, class_dir, 'train', 'image')
    train_label_dir = os.path.join(root_dir, class_dir, 'train', 'label')
    test_image_dir = os.path.join(root_dir, class_dir, 'test', 'image')
    test_label_dir = os.path.join(root_dir, class_dir, 'test', 'label')

    train_features, train_labels, train_len = load_file('train', train_image_dir, train_label_dir)
    
    test_features, test_labels, test_len = load_file('test', test_image_dir, test_label_dir)
    print('Train:', train_len)
    print('test:', test_len)

    return train_features, train_labels, test_features, test_labels


def label_indices(colormap, colormap2label):
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


class SegDataset(gdata.Dataset):
    def __init__(self, features, labels, colormap2label):
        self.rgb_mean = nd.array([0.485, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        self.features = [self.normalize_image(feature) for feature in features]
        self.labels = labels
        self.colormap2label = colormap2label

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def __getitem__(self, item):
        feature, label = self.features[item], self.labels[item]
        return feature.transpose((2, 0, 1)), label_indices(label, self.colormap2label)

    def __len__(self):
        return len(self.features)
