# from https://www.cnblogs.com/xiximayou/p/13548514.html
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSet = pd.read_csv(r'./4.0.csv')
dataSet = dataSet.values.tolist()
dataSet = np.array(dataSet)


def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S

S = calEuclidDistanceMatrix(dataSet)
print(S)

def myKNN(S, k, sigma=1.0):
    N = len(S)
    #定义邻接矩阵
    A = np.zeros((N,N))
    for i in range(N):
        #对每个样本进行编号
        dist_with_index = zip(S[i], range(N))
        #对距离进行排序
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        #取得距离该样本前k个最小距离的编号
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
        #构建邻接矩阵
        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually

    return A
A = myKNN(S,3)
print(A)

def calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

L_sys = calLaplacianMatrix(A)

#特征值分解
lam, V = np.linalg.eig(L_sys) # H'shape is n*n
lam = zip(lam, range(len(lam)))
lam = sorted(lam, key=lambda x:x[0])
H = np.vstack([V[:,i] for (v, i) in lam[:1000]]).T
H = np.asarray(H).astype(float)


from sklearn.cluster import KMeans
def spKmeans(H):
    sp_kmeans = KMeans(n_clusters=2).fit(H)
    return sp_kmeans.labels_

labels = spKmeans(H)
# plt.title('spectral cluster result')
# plt.scatter(dataSet[:, 0], dataSet[:, 1], marker='o',c=labels)
# plt.show()

pure_kmeans = KMeans(n_clusters=2).fit(dataSet)
plt.title('pure kmeans cluster result')
plt.scatter(dataSet[:, 0], dataSet[:, 1], marker='o',c=pure_kmeans.labels_)
plt.show()


# plt.title('make_circles function example')
# plt.scatter(x1[:, 0], x1[:, 1], marker='o')
# plt.show()

