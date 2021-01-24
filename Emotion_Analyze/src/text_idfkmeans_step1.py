#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
import codecs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src import tfidf_Res

# 聚类评价方法
def evaluation(tfidf_weight):
    #利用轮廓系数法选择k
    Scores = []  # 存放轮廓系数
    for k in range(3, 5):
        km = KMeans(n_clusters=k)  # 构造聚类器
        km.fit(tfidf_weight)
        Scores.append(metrics.silhouette_score(tfidf_weight, km.labels_, metric='euclidean'))
    # 求最优k值
    print(Scores)
    k = 3 + (Scores.index(max(Scores)))

    return k

# 聚类过程
def kmeans(tfidf_matrix, title_list, cluster_ResFileName):
    path2=codecs.open("./tfidf_Matrix.txt", 'w', encoding='UTF-8')
    for line in tfidf_matrix:
        path2.write("".join(str(line))+"\n")
    k = evaluation(tfidf_matrix)
    # 三、KMeans算法
    km = KMeans(n_clusters=k)
    km.fit(tfidf_matrix)
    print(Counter(km.labels_))  # 打印每个类多少个
    # 存储每个样本所属的簇
    clusterRes = codecs.open(cluster_ResFileName, 'w', encoding='UTF-8')
    count = 1
    while count <= len(km.labels_):
        clusterRes.write(str(title_list[count - 1]) + '\t' + str(km.labels_[count - 1]))
        clusterRes.write('\r\n')
        count = count + 1
    clusterRes.close()

    # # 四、可视化
    # # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(tfidf_matrix)
    x = []
    y = []
    for i in decomposition_data:
         x.append(i[0])
         y.append(i[1])

    plt.scatter(x, y, c=km.labels_)
    plt.show()
    return k
# 类似于主函数
def excute():
    # 获取TextProcess对象
    tc = tfidf_Res.TextCluster()
    data = "./get_data/data.txt"
    #这里求出了tfidf矩阵和全部的评论矩阵及词袋模型
    tfidf_matrix, title_list = tc.process(data)
    cluster_ResFileName = "cluster_idfkmResult.txt"
    k=kmeans(tfidf_matrix, title_list, cluster_ResFileName)
    print("idf方法第一阶段执行完成，下一阶段由lda方法执行")
    return k