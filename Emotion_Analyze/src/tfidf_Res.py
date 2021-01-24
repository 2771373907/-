#!/usr/bin/python
# -*- coding: UTF-8 -*-
import jieba
import logging
import numpy as np
import traceback
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from src import tfidf_implemention
import os
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
    # 提取特征
    def process(self, process_file):
        try:
            # 一、获取正文和分词
            sen_seg_list = []
            title_list = []
            flag, lines = self.load_processfile(process_file)
            if flag == False:
                logging.error("load error")
                return False, "load error"
            for line in lines:
                title_list.append(line)
                sen_seg_list.append(self.seg_words(line.split(',')[1]))
            #自己实现的tfidf
            tfidf_implemention.tfidf_Calculate(sen_seg_list)
            # 二、tf-idf提取特征
            #这里用的是sklearn的实现，放在模型里跑得更快一些.
            #自己的实现版本见ifidf_implemention,自己实现的有点笨，算的也有点慢，但效果一样
            #都是由分词文本计算出tfidf矩阵。
            # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
            tf_vectorizer = CountVectorizer()

            # fit_transform是将文本转为词频矩阵
            tf_matrix = tf_vectorizer.fit_transform(sen_seg_list)
            # 该类会统计每个词语的tf-idf权值
            tfidf_transformer = TfidfTransformer()
            np.set_printoptions(threshold=np.inf)
            # fit_transform是计算tf-idf
            tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)
            return tfidf_matrix, title_list
        except:
            logging.error(traceback.format_exc())
            return False, "process fail"
    # 释放内存资源
    def __del__(self):
        pass