import numpy as np
import math

class Hough_transform:

    def __init__(self, img, angle, step=5, threshold=135):
        '''

        :param img: 输入的图像
        :param angle: 输入的梯度方向矩阵
        :param step: Hough 变换步长大小
        :param threshold: 筛选单元的阈值
        '''
        self.img = img
        self.angle = angle
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y ** 2 + self.x ** 2))  # 向上取整
        self.step = step
        self.vote_matrix = np.zeros(
            [math.ceil(self.y / self.step), math.ceil(self.x / self.step), math.ceil(self.radius / self.step)])
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单
        元进行投票。每个点投出来结果为一折线。
        :return:  投票矩阵
        '''
        print('Hough_transform_algorithm')
        # ------------- write your code bellow ----------------
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] > 0:
                    y = i
                    x = j
                    r = 0
                    while y < self.y and x < self.x and y >= 0 and x >= 0:  # 拟合曲线，vote
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                            math.floor(r / self.step)] += 1
                        y = y + self.step * self.angle[i][j]
                        x = x + self.step
                        r = r + math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)

                    y = i - self.step * self.angle[i][j]
                    x = j - self.step
                    r = math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                            math.floor(r / self.step)] += 1
                        y = y - self.step * self.angle[i][j]
                        x = x - self.step
                        r = r + math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)

        # ------------- write your code above ----------------
        return self.vote_matrix

    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制。
        :return: None
        '''
        print('Select_Circle')

        # ------------- write your code bellow ----------------
        possible_circles = []
        for i in range(0, math.ceil(self.y / self.step)):
            for j in range(0, math.ceil(self.x / self.step)):
                for r in range(0, math.ceil(self.radius / self.step)):
                    if self.vote_matrix[i][j][r] >= self.threshold:
                        votes = self.vote_matrix[i][j][r]
                        y = i * self.step + self.step / 2
                        x = j * self.step + self.step / 2
                        r = r * self.step + self.step / 2

                        possible_circles.append((math.ceil(x), math.ceil(y), math.ceil(r), votes))
        if len(possible_circles) == 0:
            print("No Circle!")
            return
        print(possible_circles)
        possible_circles = np.array(possible_circles)
        # possible_circles.sort(key= lambda x: x[0])# 排序
        # 非极大化抑制

        x1 = possible_circles[:, 0] - possible_circles[:, 2]
        y1 = possible_circles[:, 1] - possible_circles[:, 2]
        x2 = possible_circles[:, 0] + possible_circles[:, 2]
        y2 = possible_circles[:, 1] + possible_circles[:, 2]

        # r = possible_circles[:, r]
        votes_num = possible_circles[:, 3]  # 按照票数排列

        # 外接正方形的面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # order = votes_num.argsort()[::-1]
        order = votes_num.argsort()[::-1]
        while order.size > 0:
            print(order.size)
            i = order[0]
            self.circles.append(possible_circles[i])  # 保留该类剩余box中得分最高的一个
            # 得到相交区域,左上及右下
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算相交的面积,不重叠时面积为0
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 保留IoU小于阈值的box
            inds = np.where(ovr <= 0.8)[0]  # ovr小，表示交集小，可能是两个圆形物体被检测
            order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

        # ------------- write your code above ----------------

    def Calculate(self):
        '''
        按照算法顺序调用以上成员函数
        :return: 圆形拟合结果图，圆的坐标及半径集合
        '''
        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles
