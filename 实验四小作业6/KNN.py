from numpy import *
import operator
import numpy as np
import matplotlib.pyplot as plt

def classify0(inX, dataSet, labels, k):
    '''
    KNN算法实现
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def getdata_girl_and_man(manfile,girlfile):
    '''
    用来将男生文件和女生文件合起来
    :param manfile: 男生文件名
    :param girlfile: 女生文件名
    :return: 合起来后的data,labels,列表
    '''
    f = open(manfile)
    lines = f.readlines()
    mansls = []
    for line in lines:
        a = line.split()
        a = [float(i) for i in a]
        mansls.append(a)
    manslabels = [1] * np.shape(mansls)[0]

    f1 = open(girlfile)
    lines1 = f1.readlines()
    girlls = []
    for line in lines1:
        a1 = line.split()
        a1 = [float(i) for i in a1]
        girlls.append(a1)
    girllabels = [0] * np.shape(girlls)[0]
    manslabels.extend(girllabels)
    alllabels = manslabels
    mansls.extend(girlls)
    alldatas = mansls
    return alldatas,alllabels


def getdata(filename1,classnumber):
    '''
    导入单个txt文件，将其转成列表
    :param filename1: 文件名
    :param classnumber: 将其判为哪一类，0or1
    :return: 生成好的data,labels
    '''
    f  = open(filename1)
    lines  = f.readlines()
    mansls= []
    for line in lines:
        a=line.split()
        a= [float(i) for i in a ]
        mansls.append(a)
    manslabels= [classnumber]*np.shape(mansls)[0]
    return mansls,manslabels

def draw(alldatas,k):
    '''
    做出KNN分界线
    :param alldatas: 数据集
    :param k: k近邻
    :return:
    '''
    mantest = []  # 遍历每个点，将被判为男生的加入列表，最后作图就能得到分界面
    girltest = []

    for i in np.arange(145, 185, 0.1):
        for j in np.arange(30, 90, 0.1):
            if classify0([i, j], alldatas[:, 0:2], alllabels, k) == 1:
                mantest.append([i, j])
            else:
                girltest.append([i, j])

    ax = plt.subplot(111)
    ax.scatter(np.array(mantest)[:, 0], np.array(mantest)[:, 1], s=10, c='tan', marker='s')
    ax.scatter(np.array(girltest)[:, 0], np.array(girltest)[:, 1], s=10, c='silver', marker='s')
    ax.scatter(np.array(mansls)[:, 0], np.array(mansls)[:, 1], s=30, c='red', marker='s')  # 训练数据散点图
    ax.scatter(np.array(girlsls)[:, 0], np.array(girlsls)[:, 1], s=30, c='green', marker='s')
    plt.show()


mansls,manslabels=getdata('boy.txt',0)
girlsls,girlslabels=getdata('girl.txt',1)

manslabels.extend(girlslabels)
alllabels = manslabels
mansls.extend(girlsls)
alldatas = np.array(mansls)
draw(alldatas,4)



