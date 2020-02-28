from mxnet import image, nd
import os
import numpy as np
import cv2 as cv

label_dir = os.path.join('..', 'data', 'class_05', 'label')
predict_dir = os.path.join('..', 'data', 'class_05', 'predict_label')

class_01_label_dir = os.path.join(label_dir, '01')
class_01_predict_dir = os.path.join(predict_dir, '01')

class_02_label_dir = os.path.join(label_dir, '02')
class_02_predict_dir = os.path.join(predict_dir, '02')

class_01_label_list = os.listdir(class_01_label_dir)
class_01_predict_list = os.listdir(class_01_predict_dir)
class_02_label_list = os.listdir(class_02_label_dir)
class_02_predict_list = os.listdir(class_02_predict_dir)

label_dirs = [class_01_label_dir, class_02_label_dir]
predict_dirs = [class_01_predict_dir, class_02_predict_dir]

label_lists = [class_01_label_list, class_02_label_list]
predict_lists = [class_01_predict_list, class_02_predict_list]

# print(class_01_label_list)
# print(class_01_predict_list)

idx = 1  # 第一类:0, 第二类:1


def cal_meanAccuracy(idx):
    ans = 0
    for i in range(len(label_lists[idx])):
        img_label = label_lists[idx][i]
        img_predict = predict_lists[idx][i]
        # print(img_label)
        # print(img_predict)
        img_label = cv.imread(os.path.join(label_dirs[idx], img_label), cv.IMREAD_COLOR)
        img_predict = cv.imread(os.path.join(predict_dirs[idx], img_predict), cv.IMREAD_COLOR)
        # print(img_label.shape)
        # print(img_predict.shape)
        mask = (img_predict[:, :, 0:3] == img_label[:, :, 0:3])
        img_cmp = np.zeros(shape=img_label.shape)
        img_cmp[mask] = 1
        img_cmp = img_cmp.min(axis=2)
        ans += np.sum(img_cmp)

    count = len(label_lists[idx])
    accuracy = ans / (count * 512 * 512)
    print('Mean Accuracy:', accuracy)


# 单标签IU
# def cal_mIU(idx):
#     # 注意OpenCV的读取RBG三通道顺序为：BGR！！！
#     COLOR = [[255, 255, 255], [0, 0, 255]]
#     mIU = 0.0
#     ans = 0
#     for k in range(len(label_lists[idx])):
#         img_label = label_lists[idx][k]
#         img_predict = predict_lists[idx][k]
#         img_label = cv.imread(os.path.join(label_dirs[idx], img_label), cv.IMREAD_COLOR)
#         img_predict = cv.imread(os.path.join(predict_dirs[idx], img_predict), cv.IMREAD_COLOR)
#
#         mask_label = (img_label[:, :, 0:3] == COLOR[idx])
#         img_l = np.zeros(img_label.shape)
#         img_l[mask_label] = 1
#         img_l = img_l.min(axis=2)
#         count_l = np.sum(img_l)
#
#         mask_predict = (img_predict[:, :, 0:3] == COLOR[idx])
#         img_p = np.zeros(img_predict.shape)
#         img_p[mask_predict] = 1
#         img_p = img_p.min(axis=2)
#         count_p = np.sum(img_p)
#
#         img_inter = np.zeros(img_l.shape)
#         for i in range(512):
#             for j in range(512):
#                 if (img_l[i, j] == img_p[i, j]) and (img_l[i, j] == 1):
#                     img_inter[i, j] = 1
#         count_inter = np.sum(img_inter)
#         # print(count_inter, count_l, count_p, img_inter.shape)
#         p = (count_l + count_p - count_inter)
#         if p != 0:
#             IU = count_inter / (count_l + count_p - count_inter)
#             mIU += IU
#             ans += 1
#             print(IU)
#
#     mIU = mIU / ans
#     print('Mean IU:', mIU)

# 全标签IU
def cal_mIU(idx):
    # 注意OpenCV的读取RBG三通道顺序为：BGR！！！
    COLOR = [[255, 255, 255], [0, 0, 255], [0, 0, 0]]  # 包括背景
    mIU = 0.0
    for k in range(len(label_lists[idx])):
        img_label = label_lists[idx][k]
        img_predict = predict_lists[idx][k]
        img_label = cv.imread(os.path.join(label_dirs[idx], img_label), cv.IMREAD_COLOR)
        img_predict = cv.imread(os.path.join(predict_dirs[idx], img_predict), cv.IMREAD_COLOR)
        IU = 0.0
        ans = 0  # 只计算标注图与预测图出现的标签，都没出现的标签不算
        for color_idx in range(len(COLOR)):
            mask_label = (img_label[:, :, 0:3] == COLOR[color_idx])
            img_l = np.zeros(img_label.shape)
            img_l[mask_label] = 1
            img_l = img_l.min(axis=2)
            count_l = np.sum(img_l)

            mask_predict = (img_predict[:, :, 0:3] == COLOR[color_idx])
            img_p = np.zeros(img_predict.shape)
            img_p[mask_predict] = 1
            img_p = img_p.min(axis=2)
            count_p = np.sum(img_p)

            img_inter = np.zeros(img_l.shape)
            for i in range(512):
                for j in range(512):
                    if (img_l[i, j] == img_p[i, j]) and (img_l[i, j] == 1):
                        img_inter[i, j] = 1
            count_inter = np.sum(img_inter)
            # print(count_inter, count_l, count_p, img_inter.shape)
            p = (count_l + count_p - count_inter)
            if p != 0:
                IU += count_inter / (count_l + count_p - count_inter)
                ans += 1
        IU /= ans
        mIU += IU
        print(IU)

    mIU = mIU / len(label_lists[idx])
    print('Mean IU:', mIU)


# 单标签Recall
# def cal_recall(idx):
#     ans = 0
#     mRecall = 0.0
#     # 注意OpenCV的读取RBG三通道顺序为：BGR！！！
#     COLOR = [[255, 255, 255], [0, 0, 255]]  # 包括背景
#     for k in range(len(label_lists[idx])):
#         img_label = label_lists[idx][k]
#         img_predict = predict_lists[idx][k]
#         img_label = cv.imread(os.path.join(label_dirs[idx], img_label), cv.IMREAD_COLOR)
#         img_predict = cv.imread(os.path.join(predict_dirs[idx], img_predict), cv.IMREAD_COLOR)
#
#         mask_l = (img_label[:, :, 0:3] == COLOR[idx])
#         img_l = np.zeros(img_label.shape)
#         img_l[mask_l] = 1
#         img_l = img_l.min(axis=2)
#
#         mask_p = (img_predict[:, :, 0:3] == COLOR[idx])
#         img_p = np.zeros(img_predict.shape)
#         img_p[mask_p] = 1
#         img_p = img_p.min(axis=2)
#
#         # print(img_l.shape, img_p.shape)
#         img_inter = np.zeros(img_l.shape)
#         for i in range(512):
#             for j in range(512):
#                 if img_l[i, j] == img_p[i, j] and img_l[i, j] == 1:
#                     img_inter[i, j] = 1
#
#         TP = np.sum(img_inter)
#         TP_FN = np.sum(img_l)
#         if TP_FN != 0:
#             ans += 1
#             recall = TP / TP_FN
#             mRecall += recall
#             print('Recall:', recall)
#     mRecall /= ans
#     print('Mean Recall:', mRecall)


def cal_recall(idx):
    mRecall = 0.0
    # 注意OpenCV的读取RBG三通道顺序为：BGR！！！
    COLOR = [[255, 255, 255], [0, 0, 255], [0, 0, 0]]  # 包括背景
    for k in range(len(label_lists[idx])):
        img_label = label_lists[idx][k]
        img_predict = predict_lists[idx][k]
        img_label = cv.imread(os.path.join(label_dirs[idx], img_label), cv.IMREAD_COLOR)
        img_predict = cv.imread(os.path.join(predict_dirs[idx], img_predict), cv.IMREAD_COLOR)
        recall = 0.0
        ans = 0  # 只计算单张图片中出现的标签（并非每张图都是三标签）
        for color_idx in range(len(COLOR)):
            mask_l = (img_label[:, :, 0:3] == COLOR[color_idx])
            img_l = np.zeros(img_label.shape)
            img_l[mask_l] = 1
            img_l = img_l.min(axis=2)

            mask_p = (img_predict[:, :, 0:3] == COLOR[color_idx])
            img_p = np.zeros(img_predict.shape)
            img_p[mask_p] = 1
            img_p = img_p.min(axis=2)

            # print(img_l.shape, img_p.shape)
            img_inter = np.zeros(img_l.shape)
            for i in range(512):
                for j in range(512):
                    if img_l[i, j] == img_p[i, j] and img_l[i, j] == 1:
                        img_inter[i, j] = 1

            TP = np.sum(img_inter)
            TP_FN = np.sum(img_l)
            if TP_FN != 0:
                ans += 1
                recall += TP / TP_FN

        recall /= ans
        mRecall += recall
        print('Recall:', recall)

    mRecall /= len(label_lists[idx])
    print('Mean Recall:', mRecall)


# cal_meanAccuracy(idx)
# cal_mIU(idx)
cal_recall(idx)
