import jieba
from gensim import corpora, models
import logging
import traceback
import jieba.posseg as jp
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import random

# 停用词地址
stopwords_path = "../stopwords.txt"

class TextCluster(object):
    # 初始化函数,重写父类函数
    def __init__(self, stopwords_path = stopwords_path):
        self.stopwords_path = stopwords_path

    # 分词(使用停用词)
    def seg_words(self, sentences, stopwords_path = None):
        if stopwords_path is None:
            stopwords_path = self.stopwords_path

        def stopwordslist(filepath):
            stopwords = [line.strip() for line in open(filepath, 'r', encoding='UTF-8').readlines()]
            return stopwords

        stopwords = stopwordslist(stopwords_path)  # 这里加载停用词的路径
        title_list = []
        sen_seg_list = []
        for line in sentences:
                title_list.append(line)
                flags = ('a')  # 词性
                words = [w.word for w in jp.cut(line) if w.flag in flags and w.word not in stopwords]
                sen_seg_list.append(words)
        return title_list, sen_seg_list

    # 加载用户词典
    def load_userdictfile(self, dict_file):
        jieba.load_userdict(dict_file)

    # 读取用户数据
    def load_processfile(self, process_file,i):
        corpus_list = []
        try:
            fp = open(process_file, "r", encoding='UTF-8')
            for line in fp:
                conline = line.strip()
                if conline.endswith(str(i)):
                    corpus_list.append(conline)
            return True, corpus_list
            fp.close()
        except:
            logging.error(traceback.format_exc())
            return False, "get process file fail"

    def evaluate_km(self, tfidf_weight):
        # 利用轮廓系数法选择k
        Scores = []  # 存放轮廓系数
        for k in range(2, 5):
            km = KMeans(n_clusters=k)  # 构造聚类器
            km.fit(tfidf_weight)
            Scores.append(metrics.silhouette_score(tfidf_weight, km.labels_, metric='euclidean'))
# 求最优k值
        print(Scores)
        k = 2 + (Scores.index(max(Scores)))
        return k

    # lda模型，评估num_topics设置主题的个数（聚类无需用）
    def evaluate_lda(self, corpus, dictionary):
        # shuffle corpus洗牌语料库
        cp = list(corpus)
        random.shuffle(cp)

        p = int(len(cp) * .85)
        cp_train = cp[0:p]
        cp_test = cp[p:]

        Perplex = []
        for i in range(15,25):
            lda = models.ldamodel.LdaModel(corpus=cp_train, id2word=dictionary, num_topics=i)
            # Perplexity = lda.log_perplexity(cp_test)
            perplex = lda.bound(cp_test)
            Perplexity = (np.exp2(-perplex / sum(cnt for document in cp_test for _, cnt in document)))

            Perplex.append(Perplexity)

        num_topics = 15 + (Perplex.index(min(Perplex)))
        print(Perplex)
        return num_topics

    # 释放内存资源
    def __del__(self):
        pass

    # 聚类过程
    def process(self, process_file,m):
        try:
            # 一、获取正文和分词
            flag, lines = self.load_processfile(process_file,m)
            if flag == False:
                logging.error("load error")
                return False, "load error"
            # 分词结果与其他方法形式不同
            #title_list里面存的是所有的评论
            #sen_seg_list里面存的是评论分词的列表
            title_list, sen_seg_list = self.seg_words(lines)


            # 二、lda模型提取特征
            # 构造词典
            #筛选小型心态词典
            degree = open("..\emotiondir\degree.txt", encoding='utf-8').readlines()
            negative = open("..\emotiondir\\negative.txt", encoding='utf-8').readlines()
            notwords = open("..\emotiondir\\not.txt", encoding='utf-8').readlines()
            positive = open("..\emotiondir\positive.txt", encoding='utf-8').readlines()
            isEmotion = False
            for i in sen_seg_list:
                j = 0
                while j < len(i):
                    isEmotion = False
                    if i[j] + "\n" in degree or i[j] + "\n" in negative or i[j] + "\n" in notwords or i[
                        j] + "\n" in positive:
                        isEmotion = True
                    if isEmotion == False:
                        i.remove(i[j])
                        j = j - 1
                    j = j + 1
            i = 0
            while i < len(sen_seg_list):
                if len(sen_seg_list[i]) == 0:
                    sen_seg_list.remove(sen_seg_list[i])
                    i = i - 1
                i = i + 1
            dictionary = corpora.Dictionary(sen_seg_list)
            # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
            corpus = [dictionary.doc2bow(words) for words in sen_seg_list]

            lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=1)
            # 打印所有主题，每个主题显示10个词
            for topic in lda.print_topics(num_words=10):
                print("第{0}类的关键词为：".format(m))
                print(topic)
        except:
            logging.error(traceback.format_exc())
            return False, "process fail"
    # 获取TextProcess对象
def excute(m):
    tc = TextCluster(stopwords_path)
    data = "cluster_ldakmResult.txt"
    k=tc.process(data,m)
    return k

