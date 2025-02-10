
# from https://blog.csdn.net/weixin_46713695/article/details/125667255

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')


def dist(node1, node2):
    """
    计算欧几里得距离,node1,node2分别为两个元组
    """
    return math.sqrt(math.pow(node1[0]-node2[0],2)+math.pow(node1[1]-node2[1],2))


# DBSCAN算法，参数为数据集，Eps为指定半径参数，MinPts为制定邻域密度阈值
def dbscan(Data, Eps, MinPts):
    num = len(Data)  # 点的个数
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]  # C为输出结果，默认是一个长度为num的值全为-1的列表        
    k = -1  # 用k来标记不同的簇，k = -1表示噪声点
    while len(unvisited) > 0:        
        p = random.choice(unvisited)  # 随机选择一个unvisited对象
        unvisited.remove(p)
        visited.append(p)        
        N = []  # N为p的epsilon邻域中的对象的集合
        for i in range(num):
            if (dist(Data[i], Data[p]) <= Eps):  # and (i!=p):
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k + 1
            C[p] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if (dist(Data[j], Data[pi]) <= Eps):  
                            M.append(j)
                    if len(M) >= MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1
    return C

dataSet = pd.read_csv(r'./4.0.csv')
dataSet = dataSet.values.tolist()
print(dataSet)
print(type(dataSet))
C = dbscan(dataSet, 0.100001, 1)
x, y = [], []
for data in dataSet:
    x.append(data[0])
    y.append(data[1])
print(C)
# print(type(dataSet))
plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(x, y, c=C, marker='o')
plt.show()

#评价:
#有瑕疵:不能预先指定聚类数量，属于看似效果很好的聚类算法。