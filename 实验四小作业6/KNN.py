from numpy import *
import operator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def classify0(inX, dataSet, labels, k):
    '''
    KNN算法实现
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    '''
    result=k
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
    # print(sortedClassCount)
    if len(sortedClassCount)>1:
        result=sortedClassCount[0][1]-sortedClassCount[1][1]
    return sortedClassCount[0][0],result

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

def draw(alldatas,alllabels,mansls,girlsls,k):
    '''
    做出KNN分界线
    只需要判断一个点被判为0或者1的次数是否相等就可以，相等就是分界面的一员
    :param alldatas: 数据集
    :param k: k近邻
    :return:
    '''
    fenjiemian=[]

    for i in tqdm(np.arange(np.mean(alldatas[:,0])-30, np.mean(alldatas[:,0])+20, 6)):#步长放长，避免过拟合，分界面更平滑
        for j in np.arange(np.mean(alldatas[:,1])-30, np.mean(alldatas[:,1])+20, 6):
            for z in np.arange(np.mean(alldatas[:,2])-10, np.mean(alldatas[:,2])+10, 6):
                class_result,class_sub=classify0([i, j, z], alldatas[:, 0:3], alllabels, k)
                if class_sub==0 :
                    fenjiemian.append([i, j,z])

    ax = plt.subplot(111,projection='3d')
    ax.plot_trisurf(np.array(fenjiemian)[:, 0], np.array(fenjiemian)[:, 1],np.array(fenjiemian)[:, 2])#画出分界面
    ax.scatter(np.array(mansls)[:, 0], np.array(mansls)[:, 1], np.array(mansls)[:, 2], c='red')
    ax.scatter(np.array(girlsls)[:, 0], np.array(girlsls)[:, 1], np.array(girlsls)[:, 2], c='green')
    plt.show()

#
mansls1,manslabels1=getdata('boy.txt',0)
girlsls1,girlslabels1=getdata('girl.txt',1)
m=mansls1.copy()
g=girlsls1.copy()
manslabels1.extend(girlslabels1)
alllabels = manslabels1
mansls1.extend(girlsls1)
alldatas = np.array(mansls1)
draw(alldatas,alllabels,m,g,6)
print(np.mean(alldatas[:,0]))

# print(print('\n'.join([''.join([('LoveAndy'[(x-y)%8]if((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3<=0 else' ')for x in range(-30,30)])for y in range(15,-15,-1)])))



