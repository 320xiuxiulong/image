# _*_ coding:UTF-8 _*_
# @TIME:2019/9/25 18:21
import cv2
from pylab import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 从图像文件中读取图像数据并显示
file = '11.jpg'
grayFile = 'test.jpg'
# cv2.namedWindow("Image")
img = cv2.imread(file)
cv2.imshow("img", img)
# 把彩色图像转化成灰度图像并显示
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('test.jpg', imgray, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
cv2.imshow("imgray", imgray)
# 在灰度图像中添加几何形状并显示
imgray1 = cv2.imread(grayFile, 1)
print(np.shape(imgray1))
# 画一个正方形
for i in range(10, 20):
    for j in range(10, 20):
        imgray1[i, j] = (255, 0, 0)
# 画一条直线
for i in range(30, 50):
    for j in range(10, 12):
        imgray1[i, j] = (0, 0, 255)
# 画一个三角形
for i in range(40, 50):
    for j in range(40, i):
        imgray1[i, j] = (0, 255, 0)

cv2.imshow('draw', imgray1)
cv2.imwrite('test.jpg', imgray1, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
h, w, _ = imgray1.shape
pixel = []
for i in range(h):
    for j in range(w):
        pixel.append((i, j, imgray[i][j]))
data = np.array(pixel).reshape((h * w, 3))

cv2.waitKey(0)
cv2.destroyWindow("Image")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, m, zlow, zhigh in [('r', '.', -200, 300)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
