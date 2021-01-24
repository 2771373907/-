#输入：分好词的sen_seg_list
#返回值：tfidf_matrix矩阵。
import codecs
import math


def Counter(word_list):
    wordcount=[]
    for i in word_list:
        count={}
        for j in i:
            if not count.get(j):
                count.update({j:1})
            elif count.get(j):
                count[j]+=1
        wordcount.append(count)
    return wordcount
def tf(word,count):
    return count[word]/sum(count.values())
def n_containing(word,wordcount):
    return sum(1 for i in wordcount if i.get(word))
def idf(word,wordcount):
    return math.log(len(wordcount)/(1+n_containing(word,wordcount)))
def tfidf(word,word_list,wordcount):
    return tf(word,word_list)*idf(word,wordcount)
def tfidf_Calculate(sen_seg_list):
    Res = codecs.open("./FinalTFIDFValue.txt", 'w', encoding='UTF-8')
    words=[]
    for i in sen_seg_list:
        words.append(i.split())
    wordcount=Counter(words)
    res=sen_seg_list
    temp={}
    p=1
    for i in wordcount:
        print("part:{}".format(p))
        p=p+1
        for j,k in i.items():
            value=tfidf(j,i,wordcount)
            print("{}----TF-IDF:{}".format(j,value))
            Res.write("{}----TF-IDF:{}".format(j,value))
            if not temp:
                temp[j]=value
            else:
                needadd=False
                for k,m in temp.items():
                    if m<value and len(temp)<100:
                        needadd=True
                        break
                    if m<value and len(temp)==100:
                        del temp[k]
                        needadd=True
                        break
                if needadd==True:
                    temp[j]=value
    keyemotion={}
    coreemotion={}
    #得到了关键词之后与心态词碰撞。
    degree = open("..\emotiondir\degree.txt", encoding='utf-8').readlines()
    negative = open("..\emotiondir\\negative.txt", encoding='utf-8').readlines()
    notwords = open("..\emotiondir\\not.txt", encoding='utf-8').readlines()
    positive = open("..\emotiondir\positive.txt", encoding='utf-8').readlines()
    DIYEmotionDir=open("..\emotiondir\DIYEmotionDir.txt",encoding='utf-8').readlines()
    for key,value in temp.items():
        if key+"\n" in degree or key+"\n" in negative or key+"\n" in notwords or key+"\n" in positive:
            keyemotion[key]=value
        if key+"\n" in DIYEmotionDir:
            coreemotion[key]=value
    print("标准心态词典筛选关键情绪词为：")
    print(keyemotion)
    print("自定义心态词典筛选核心情绪词为：")
    print(coreemotion)
    print("第二种tfidf实现完成")


