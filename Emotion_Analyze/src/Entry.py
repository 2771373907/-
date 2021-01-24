#共有三种方法：
#1.tfidf方法
#2.lda方法
#3.doc2veck方法
from src.get_data import get_res_csv
from src import text_idfkmeans_step2
from src import text_ldakmeans_step1
from src import text_ldakmeans_step2
from src import text_idfkmeans_step1
from src import text_doc2veckmeans_step1
from src import text_doc2veckmeans_step2
from src import finalMethod
#step1,2:对样本进行处理，用心态词典做关键词，统计既在心态词典中的词，又在关键词中的词。(直接调用3组Method方法，处理样本集）
#step2:对总体进行处理，统计上一部分词的重要程度。（调用FinalMethod方法，处理总体集）
#花边：1.对总体进行聚类分类，并求出每一类的主要心态。（直接调用3组Method方法，处理总体集）
#花边：2.对分类过程进行可视化（调用方法的同时就弹出可视化表格了）
#花边：3.亮点：tfidf方法和lda方法，doc2vec方法，k聚类的代码级实现（三个文档分别是分类之后的结果，跑三个方法会得出每一类的关键词。）。
#注：finalMethod使用自己建的小心态词典，类似ldaMethod_step2做的。
#小心态词典建立的原则：从lda方法中选取各个主题前10关键词中的心态词，并从doc2vec和tfidf得出的各类关键词中各补充10个心态词。
# 建立新的心态词典，使用lda方法计算各个心态词的重要程度（调用finalMethod方法）。
# 对于原始心态词典，我们读了百分之十的评论，加入了一些关键词
def Getcsv():
    get_res_csv.excute()
def TfidfMethod():
    k=text_idfkmeans_step1.excute()
    for i in range(0,k):
        text_idfkmeans_step2.excute(i)
def LdaMethod():
    k=text_ldakmeans_step1.excute()
    for i in range(0,k):
        text_ldakmeans_step2.excute(i)
def Doc2vectorMethod():
    #cluster_ldakmResult是text_ldakmeans_step的生成文件。
    #cluster_ldakmResult_temp1到cluster_ldakmResult_tempn是将文件分类保存后的文件
    k=text_doc2veckmeans_step1.excute()
    for i in range(0,k):
        text_doc2veckmeans_step2.excute(i)
Getcsv()
TfidfMethod()
#Doc2vectorMethod()
#LdaMethod()
#finalMethod.excute("./get_data/data.txt")
