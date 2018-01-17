import os
import sys
import numpy as np
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import sympy
import byesclassify
import pandas as pd


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
    manslabels= [1]*np.shape(mansls)[0]
    return mansls,manslabels


def drawcompare(w, mean1,mean2,group1, group2,rawboy,rawgirl,xianyan1=0.5,xianyan2=0.5):
    '''
    只针对男女两类作图，针对作业，无广泛性
    :param w: LDA最终得到的投影向量
    :param mean1: 第一类的均值
    :param mean2: 第二类的均值
    :param group1: 经过转置处理后的适用于LDA的第一类的数据
    :param group2: 经过转置处理后的适用于LDA的第二类的数据
    :param rawboy:  未经处理的第一类数据用于贝叶斯
    :param rawgirl: 未经处理的第二类数据用于贝叶斯
    :param xianyan1: 贝叶斯需要用到的第一类先验概率
    :param xianyan2: 贝叶斯需要用到的第二类先验概率
    :return: 无，直接做图
    '''
    x3 = []
    y3 = []

    for i in np.arange(100, 200,0.1):
        for j in range(30, 100):
            prediction = byesclassify.duoweiclassifier([i, j], rawboy,rawgirl, xianyan1, xianyan2)
            if 0.500 == round(prediction, 3):
                x3.append(i)
                y3.append(j)
    plt.figure(1)
    ax = plt.subplot(111)
    plt.xlim(140, 200)
    plt.ylim(35, 100)
    ax.scatter(np.array(group1)[0], np.array(group1)[1], s=30, c='red', marker='s')  # 训练数据散点图
    ax.scatter(np.array(group2)[0], np.array(group2)[1], s=30, c='green')
    w = np.array(w)
    x = arange(100, 200, 1)

    y = array((w[1] * x) / w[0])  # 投影直线

    y1 = array(-w[0] * (x - 0.5 * (array(mean1)[0, 0] + array(mean2)[0, 0])) / (w[1]) + 0.5 * (
    array(mean1)[1, 0] + array(mean2)[1, 0]))  # 边界分割线
    plt.figure(1)
    ax1 = plt.subplot(111)
    # plt.plot(x,y)
    plt.plot(x, y1)
    plt.plot(x, y)
    plt.plot(x3,y3)

    # plt.plot(x,y1)
    plt.show()


def draw(w, mean1,mean2,group1, group2):
    '''
    LDA分界曲线
    :param w: 训练得到的权重
    :param mean1: 第一类样本均值
    :param mean2: 第二类样本均值
    :param group1: 第一类样本
    :param group2: 第二类样本
    :return: 无，直接做图
    '''
    plt.figure(1)
    ax = plt.subplot(111)
    #ax.plot(range(0,200),range(0,200))
    plt.xlim(0,200,5)
    plt.ylim(0,200,5)
    ax.scatter(np.array(group1)[0], np.array(group1)[1], s=30, c='red', marker='s')  # 训练数据散点图
    ax.scatter(np.array(group2)[0], np.array(group2)[1], s=30, c='green')
    w = np.array(w)
    x = arange(100, 200, 1)
    y = array((w[1] * x) / w[0])  # 投影直线

    y1 = array(-w[0] * (x - 0.5 * (array(mean1)[0, 0] + array(mean2)[0, 0])) / (w[1]) + 0.5 * (
    array(mean1)[1, 0] + array(mean2)[1, 0]))  # 边界分割线
    plt.figure(1)
    ax1 = plt.subplot(111)
    # print([1]*np.shape(dot(w.T,np.array(group1)))[1])
    # ax1.scatter(dot(w.T,np.array(group1)),[1]*np.shape(dot(w.T,np.array(group1)))[1],s=30,c='blue')
    # ax1.scatter(dot(w.T,np.array(group2)),[1]*np.shape(dot(w.T,np.array(group2)))[1],s=30,c='blue')

    # plt.plot(x,y)
    plt.plot(x, y1)
    plt.plot(x,y)
    # plt.plot(x,y1)
    plt.show()

# 计算样本均值
# 参数samples为nxm维矩阵，其中n表示维数，m表示样本个数
def compute_mean(samples):
    mean_mat = mean(samples, axis=1)
    return mean_mat

def compute_withinclass_scatter(samples, mean):
    '''
    计算样本内离散度
    :param samples: 样本向量矩阵，大小为nxm，其中n表示维数，m表示样本个数
    :param mean: 表示均值向量，大小为1xd，d表示维数，大小与样本维数相同，即d=m
    :return:
    '''
    # 获取样本维数，样本个数
    dimens, nums = samples.shape[:2]
    # 将所有样本向量减去均值向量
    samples_mean = samples - mean
    # 初始化类内离散度矩阵
    s_in = 0
    for i in range(nums):
        x = samples_mean[:, i]
        s_in += dot(x, x.T)
        # endfor
    return s_in


def train(groupboy,groupgirl):
    '''
    两类LDA投影向量训练

    :param groupboy:第一类样本
    :param groupgirl: 第二类样本
    :return: 投影向量，两类均值，转置后的原两类样本
    '''

    group1=mat(np.array(groupboy)[:,0:2].T)
    group2=mat(np.array(groupgirl)[:,0:2].T)

    mean1 = compute_mean(group1)
    mean2 = compute_mean(group2)

    s_in1 = compute_withinclass_scatter(group1, mean1)
    s_in2 = compute_withinclass_scatter(group2, mean2)

    s = s_in1 + s_in2#求总类内离散度矩阵

    s_t = s.I # 求s的逆矩阵

    w = dot(s_t, mean1 - mean2)# 求解权向量
    return w,mean1,mean2,group1,group2


def predict(data, w, mean1, mean2):
    '''
    预测
    :param data:测试值
    :param w: 权值（投影向量）
    :param mean1: 第一类均值
    :param mean2: 第二类均值
    :return: 分类结果，0为阈值
    '''
    test1 = mat(data)
    g = dot(w.T, test1.T - 0.5 * (mean1 + mean2))
    return g

def to_datafram(data,label):
    '''
    将输入的txt文档列表变成dataframe，便于留一法的统计处理

    :param data: 文档属性列表
    :param label: 文档标志列表
    :return: 添加好的dataframe
    '''
    dfdata=pd.DataFrame()
    dfdata['height'] = np.array(data)[:, 0]
    dfdata['weight'] = np.array(data)[:, 1]
    dfdata['shoes_size'] = np.array(data)[:, 2]
    dfdata['labels']=label[:]
    return dfdata

def liuyi(df,feture1,feture2):
    '''
    留一法的实现

    :param df:传入的数据集
    :param feture1: 选择测试的特征1
    :param feture2: 选择测试的特征2
    :return: 留一法选择每个样本作为测试样本所得到的预测结果列表
    '''
    result=[]
    for i in range(df.shape[0]):
        test=df[[feture1, feture2]][i:i+1]
        groupboy = np.array(df[[feture1, feture2]][df['labels'] == 0][0:i]).tolist()
        groupboy.extend(np.array(df[[feture1, feture2]][df['labels']==0][i+1:]).tolist())
        groupgirl = np.array(df[[feture1, feture2]][df['labels'] == 1][0:i]).tolist()
        groupgirl.extend(np.array(df[[feture1, feture2]][df['labels'] == 1][i + 1:]).tolist())
        w, mean1, mean2, group1, group2 = train(groupboy, groupgirl)
        if(predict(np.array(test), w, mean1, mean2)[0,0]>0):
            result.append(0)
        else:
            result.append(1)
    return result

def accurcy(predict,test):
    count = 0
    n = np.shape(test)[0]
    for i in range(n) :
        if predict[i]==test[i]:
            count = count+1
    return count/n
#------------------测试LDA----------------------------
# groupboy,l=getdata('boy.txt',0)
# groupgirl,ls =getdata('girl.txt',1)
# w,mean1,mean2,group1,group2= train(groupboy,groupgirl)
# # drawcompare(w,mean1,mean2,group1,group2,groupboy,groupgirl)//贝叶斯和LDA决策面
# testboy,labels=getdata('boy82.txt',0)
# #
# group3=mat(np.array(testboy)[:,0:2].T)
# print(group3[:,1].T)
# draw(w,mean1,mean2,group3,group2)
# count=0
# print('1:',(predict(group3[:,1].T,w,mean1,mean2)))
# for i in range(shape(group3)[1]):
#     if (predict(group3[:,i].T,w,mean1,mean2))[0,0]>=0:#这里要注意和测试数据对应
#         count = count+1
# print(count/shape(group3)[1])
#----------------------------------------------------------------------
#--------------------留一法测试-----------------------------------
mandata,manlabel=getdata('boy.txt',0)
girldata,girllabel=getdata('girl.txt',1)
a,l=getdata_girl_and_man('boy.txt','girl.txt')
d=to_datafram(a,l)
print(accurcy(np.array(d['labels']).tolist(),liuyi(d,'height','shoes_size')))#height与shoes_size和weight与shoes_size效果准确率最高
#-------------------------------------------------------------------------------------------------------------


