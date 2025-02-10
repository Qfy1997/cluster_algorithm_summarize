# from https://zhuanlan.zhihu.com/p/367956614
import math
import matplotlib.pyplot as plt 


def dist(node1, node2):
    """
    计算欧几里得距离,node1,node2分别为两个元组
    """
    return math.sqrt(math.pow(node1[0]-node2[0],2)+math.pow(node1[1]-node2[1],2))

def dist_min(cluster_x, cluster_y):
    """
    取两个集合中距离最近的两个样本的距离作为这两个集合的距离
    """
    return min(dist(node1,node2) for node1 in cluster_x for node2 in cluster_y)

def dist_max(cluster_x, cluster_y):
    """
    取两个集合中距离最远的两个样本的距离作为两个集合的距离
    """
    return max(dist(node1,node2) for node1 in cluster_x for node2 in cluster_y)

def dist_avg(cluster_x, cluster_y):
    """
    先求两个集合中的点的两两距离，全部求和后取平均值
    """
    return sum(dist(node1,node2) for node1 in cluster_x for node2 in cluster_y)/(len(cluster_x)*len(cluster_y))

def find_min(distance_matrix):
    """
    找出距离最近的两个簇的下标
    """
    minvalue = 1000
    x = 0
    y = 0
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j  and distance_matrix[i][j] < minvalue :
                minvalue = distance_matrix[i][j]
                x = i
                y = j
    return (x,y,minvalue)

def find_max(distance_matrix):
    """
    找出距离最远的两个簇的下标
    """
    maxvalue = 0
    x = 0
    y = 0
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j  and distance_matrix[i][j] > maxvalue :
                maxvalue = distance_matrix[i][j]
                x = i
                y = j
    return (x,y,maxvalue)

def AGNES(dataset, distance_method,find_method,k):
    """
    层次聚类(自底向上)算法模型
    """
    # print(len(dataset))
    # 初始化簇类集合和距离矩阵
    cluster_set=[]
    distance_matrix = []
    for node in dataset:
        cluster_set.append([node])
    # print("original cluster set:")
    # print(cluster_set)
    for cluster_x in cluster_set:
        distance_list = []
        for cluster_y in cluster_set:
            distance_list.append(distance_method(cluster_x,cluster_y))
        distance_matrix.append(distance_list)
    # print("original distance matrix:")
    # print(len(distance_matrix))
    # print(len(distance_matrix[0]))
    # print(distance_matrix)
    q = len(dataset)
    # 合并更新
    while q > k:
        idx,idy,min_distance = find_method(distance_matrix)
        cluster_set[idx].extend(cluster_set[idy])
        cluster_set.remove(cluster_set[idy])
        distance_matrix = []
        for cluster_x in cluster_set:
            distance_list = []
            for cluster_y in cluster_set:
                distance_list.append(distance_method(cluster_x,cluster_y))
            distance_matrix.append(distance_list)
        q -=1
    return cluster_set

if __name__=="__main__":
    file_path="./4.0.csv"
    with open(file_path,'r') as f:
        file_data = f.readlines()
    data=[]
    for item in file_data:
        new_item=item.strip().split(',')
        ite = (float(new_item[0]),float(new_item[1]))
        data.append(ite)
    cluster_set = AGNES(data,dist_min,find_min,3)
    print("final results:")
    print("================")
    print(cluster_set[0])
    print(cluster_set[1])
    print(cluster_set[2])
    label1_x=[]
    label1_y=[]
    label2_x=[]
    label2_y=[]
    label3_x=[]
    label3_y=[]
    for item in cluster_set[0]:
        label1_x.append(item[0])
        label1_y.append(item[1])
    for item in cluster_set[1]:
        label2_x.append(item[0])
        label2_y.append(item[1])
    for item in cluster_set[2]:
        label3_x.append(item[0])
        label3_y.append(item[1])

    plt.scatter(label1_x, label1_y, c="red", marker='o', label='category0')  
    plt.scatter(label2_x, label2_y, c="green", marker='*', label='category1')  
    plt.scatter(label3_x, label3_y, c="blue", marker='+', label='category2')  
    plt.xlabel('densities')  
    plt.ylabel('sugar content')  
    plt.legend(loc=2)  
    plt.title("dist_min and find_min")
    plt.savefig("dist_min_and_find_min.png")
    plt.show()
    
