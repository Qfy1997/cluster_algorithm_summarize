# from https://www.cnblogs.com/shenfeng/p/kmeans_demo.html
# from https://mubaris.com/posts/kmeans-clustering/
from sklearn.cluster import KMeans
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# 导入数据集
# data = pd.read_csv('4.0.csv')
# print(data.shape)
# data.head()
dataSet = pd.read_csv(r'./4.0.csv')
dataSet = dataSet.values.tolist()
print(dataSet)
print(type(dataSet))
f1x, f2y = [], []
for data in dataSet:
    f1x.append(data[0])
    f2y.append(data[1])

f1 = np.array(f1x)
f2 = np.array(f2y)

X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=6)
# plt.show()

# 按行的方式计算两个坐标点之间的距离
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# 设定分区数
k = 3
# 随机获得中心点的X轴坐标
C_x = np.random.uniform(low=np.min(X), high=np.max(X), size=k)
# 随机获得中心点的Y轴坐标
C_y = np.random.uniform(low=np.min(X), high=np.max(X), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C_x)
print(C_y)
print(C)
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
# plt.show()

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
print(C)
print(centroids)

