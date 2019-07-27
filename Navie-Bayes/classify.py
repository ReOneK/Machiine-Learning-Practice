import math

import numpy as np


# 第一步，将文本转化为词向量
def createVocList(dataset):
    """
    找出所有出现的单词
    :param dataset:总的数据集
    :return: 所有出现过的单词
    """
    voc = set([])
    for doc in dataset:
        voc = doc | set(doc)
    return list(voc)


def set_word_Vec(voc, input):
    """
    将统计输入文档的字符是否出现在词集中
    :param voc: 词集
    :param input: 需要输入数据集
    :return: 出现的单词列表
    """
    voc_list = len(voc) * [0]
    for word in input:
        if word in voc:
            voc_list[voc_list.index(word)] = 1
        else:
            pass
    return voc_list


def train(trainMax, trainLabel):
    """
    训练函数
    :param trainMax: 文件单词矩阵
    :param trainLabel: 文件对应的类别
    :return:
    """
    numoftrain = len(trainMax)
    numofwords = len(trainMax[0])

    P_Ci = sum(trainLabel) / numoftrain

    p0Num = np.ones(numofwords)
    p1Num = np.ones(numofwords)

    p0_sum = 2
    p1_sum = 2

    for doc in range(trainMax):
        if trainLabel[doc] == 1:
            p0_sum += 1
            p0_sum += sum(trainLabel[doc])
        else:
            p1Num += 1
            p1_sum += sum(trainLabel[doc])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    p1Vect = math.log(p1Num / p1_sum)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = math.log(p0Num / p0_sum)
    return p0Vect, p1Vect, P_Ci


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    使用算法：
    # 将乘法转换为加法
    乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
    加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    # 计算公式 log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 大家可能会发现，上面的计算公式，没有除以贝叶斯准则的公式的分母，也就是 P(w) （P(w) 指的是此文档在所有的文档中出现的概率）就进行概率大小的比
    # 因为 P(w) 针对的是包含侮辱和非侮辱的全部文档，所以 P(w) 是相同的。
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # 我的理解是：这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)  # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)  # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1 > p0:
        return 1
    else:
        return 0


def load_data_set():
    """
    创建数据集,都是假的 fake data set
    :return: 单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec


def testingNB():
    """
    测试朴素贝叶斯算法
    """
    # 1. 加载数据集
    listOPosts, listClasses = load_data_set()
    # 2. 创建单词集合
    myVocabList = createVocList(listOPosts)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
    # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(set_word_Vec(myVocabList, postinDoc))
    # 4. 训练数据
    p0V, p1V, pAb = train(np.array(trainMat), np.array(listClasses))
    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(set_word_Vec(myVocabList, testEntry))
    print (testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(set_word_Vec(myVocabList, testEntry))
    print (testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
