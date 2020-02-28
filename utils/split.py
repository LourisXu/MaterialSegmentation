import shutil
import os
import random


def move(img_list, str, src, dst):
    for i in range(len(img_list)):
        shutil.copy(os.path.join(src, img_list[i]), os.path.join(dst, img_list[i]))
        print("Copy the %s (%s) from %s to %s" % (str, img_list[i], src, dst))


if __name__ == '__main__':
    data_dir = os.path.join('..', 'data')
    class_dir = os.path.join('class_06')
    image_dir = os.path.join(data_dir, class_dir, 'total', 'image')
    label_dir = os.path.join(data_dir, class_dir, 'total', 'label')

    train_image_dir = os.path.join(data_dir, class_dir, 'train', 'image')
    train_label_dir = os.path.join(data_dir, class_dir, 'train', 'label')
    test_image_dir = os.path.join(data_dir, class_dir, 'test', 'image')
    test_label_dir = os.path.join(data_dir, class_dir, 'test', 'label')

    image_list = os.listdir(image_dir)
    random.shuffle(image_list)
    PARTS = 5
    length = len(image_list)
    train_list = image_list[:length - length // PARTS]
    test_list = image_list[length - length // PARTS:]
    print("train: ", len(train_list))
    print("test: ", len(test_list))
    move(train_list, 'trian_img', image_dir, train_image_dir)
    move(train_list, 'train_label', label_dir, train_label_dir)
    move(test_list, 'test_img', image_dir, test_image_dir)
    move(test_list, 'test_label', label_dir, test_label_dir)

