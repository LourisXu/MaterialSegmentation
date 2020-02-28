import cv2 as cv
import os

def cal_Area(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    w, h = gray.shape
    count = sum(gray[gray == 255]) / 255
    return count / (w * h)


def calLikeCircle(img):
    R = []
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # 找到图片轮廓

    for i, contour in enumerate(contours):  # 遍历各个轮廓
        area = cv.contourArea(contour)
        (x, y), radius = cv.minEnclosingCircle(contour)
        R.append([area, radius])
    return R


data_dir = os.path.join('..', 'data', 'W800_147', 'label_png')
img_list = os.listdir(data_dir)
for i in range(len(img_list)):
    image = cv.imread(os.path.join(data_dir, img_list[i]))
    r1 = cal_Area(image)
    r2 = calLikeCircle(image)
    with open(os.path.join(data_dir, '%s.txt' % img_list[i][:-10]), 'w') as f:
        print(r1, file=f)
        for line in r2:
            print('%f %f' % (line[0], line[1]), file=f)
