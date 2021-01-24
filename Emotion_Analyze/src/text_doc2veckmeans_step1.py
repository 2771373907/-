# -*- coding: UTF-8 -*-
import jieba
import logging
import codecs
import traceback
import gensim
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
import os
import matplotlib.pyplot as plt
# 停用词地址
stopwords_path = "../stopwords.txt"

class TextCluster(object):
    # 初始化函数,重写父类函数
    def __init__(self, stopwords_path = stopwords_path):
        self.stopwords_path = stopwords_path
    # 分词(使用停用词)
    def seg_words(self, sentence, stopwords_path = None):
        if stopwords_path is None:
            stopwords_path = self.stopwords_path

        def stopwordslist(filepath):
            stopwords = [line.strip() for line in open(filepath, 'r', encoding='UTF-8').readlines()]
            return stopwords

        sentence_seged = jieba.cut(sentence.strip())
        stopwords = stopwordslist(stopwords_path)  # 这里加载停用词的路径
        outstr = ''  # 返回值是字符串
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

    # 加载用户词典
    def load_userdictfile(self, dict_file):
        jieba.load_userdict(dict_file)

    # 读取用户数据
    def load_processfile(self, process_file):
        corpus_list = []
        try:
            fp = open(process_file, "r", encoding='UTF-8')
            for line in fp:
                conline = line.strip()
                corpus_list.append(conline)
            fp.close()
            return True, corpus_list
        except:
            logging.error(traceback.format_exc())
            return False, "get process file fail"

    # 存储分词文本（不含标题）
    def output_file(self, out_file, sen_seg_list):

        try:
            data1 = codecs.open(out_file, 'w', encoding='UTF-8')
            for num in range(len(sen_seg_list)):
                if num == len(sen_seg_list) - 1:
                    data1.write(sen_seg_list[num])
                else:
                    data1.write(sen_seg_list[num] + '\r\n')
            data1.close()
        except:
            logging.error(traceback.format_exc())
            return False, "out file fail"

    # 评价算法好坏
    def evaluation(self, tfidf_matrix):
        # 利用轮廓系数法选择k
        Scores = []  # 存放轮廓系数
        for k in range(3, 5):
            km = KMeans(n_clusters=k)  # 构造聚类器
            km.fit(tfidf_matrix)
            Scores.append(metrics.silhouette_score(tfidf_matrix, km.labels_, metric='euclidean'))
        print(Scores)
        k = 3 + (Scores.index(max(Scores)))

        return k

    # 释放内存资源
    def __del__(self):
        pass

    # 聚类过程
    def process(self, process_file, num_clusters, cluster_ResFileName, data1, modelpath):
        try:
            # 一、获取标题和分词
            sen_seg_list = []
            title_list = []
            flag, lines = self.load_processfile(process_file)
            if flag == False:
                logging.error("load error")
                return False, "load error"
            for line in lines:
                title_list.append(line)
                sen_seg_list.append(self.seg_words(line.split(',')[1]))

            if not os.path.exists(modelpath):
                # 存储分词文本
                if not os.path.exists(data1):
                    self.output_file(data1, sen_seg_list)
                    print("success output")

            # doc2vec提取特征
            sentences = gensim.models.doc2vec.TaggedLineDocument(data1)

            if not os.path.exists(modelpath):
                # doc2vec提取特征
                # 训练并保存模型
                model = gensim.models.Doc2Vec(sentences, size=100, window=2, min_count=3)
                model.train(sentences, total_examples=model.corpus_count, epochs=1000)
                model.save(modelpath)

            infered_vectors_list = []
            print("load doc2vec model...")
            model_dm = gensim.models.Doc2Vec.load(modelpath)
            print("load train vectors...")
            i = 0
            for text, label in sentences:
                vector = model_dm.infer_vector(text)
                infered_vectors_list.append(vector)
                i += 1

            k = self.evaluation(infered_vectors_list)
            # Kmeans,大数据量下用Mini-Batch-KMeans算法
            km = KMeans(n_clusters=k)
            km.fit(infered_vectors_list)
            print(Counter(km.labels_))  # 打印每个类多少个

            # 存储每个样本所属的簇
            clusterRes = codecs.open(cluster_ResFileName, 'w', encoding='UTF-8')
            count = 1
            while count <= len(km.labels_):
                clusterRes.write(str(title_list[count - 1]) + '\t' + str(km.labels_[count - 1]))
                clusterRes.write('\r\n')
                count = count + 1
            clusterRes.close()
            return k

        except:
            logging.error(traceback.format_exc())
            return False, "process fail"


# 类似于主函数
def excute():
    # 获取TextProcess对象
    tc = TextCluster(stopwords_path)

    data = "./get_data/data.txt"
    k=tc.process(data, 3, "cluster_dockmResult.txt", "get_data/data1tag.txt", "./Doc2Vecmodel.pkl")
    print("success")
    return k