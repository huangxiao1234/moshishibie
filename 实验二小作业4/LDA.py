import os
import sys
import numpy as np
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import sympy
import byesclassify



def getdata(filename):
    f  = open(filename)
    lines  = f.readlines()
    mansls= []
    for line in lines:
        a=line.split()
        a= [float(i) for i in a ]
        mansls.append(a)
    # labels = np.zeros(np.shape(ls)[0])
    manslabels= [1]*np.shape(mansls)[0]
    return mansls,manslabels
# end of createDataSet


def drawcompare(w, mean1,mean2,group1, group2,rawboy,rawgirl,xianyan1=0.5,xianyan2=0.5):
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
    # ax.plot(range(0,200),range(0,200))
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


# 画图
def draw(w, mean1,mean2,group1, group2):
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

# end of draw

# 计算样本均值
# 参数samples为nxm维矩阵，其中n表示维数，m表示样本个数
def compute_mean(samples):
    mean_mat = mean(samples, axis=1)
    return mean_mat


# end of compute_mean

# 计算样本类内离散度
# 参数samples表示样本向量矩阵，大小为nxm，其中n表示维数，m表示样本个数
# 参数mean表示均值向量，大小为1xd，d表示维数，大小与样本维数相同，即d=m
def compute_withinclass_scatter(samples, mean):
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


# end of compute_mean
def train(groupboy,groupgirl):

    group1=mat(np.array(groupboy)[:,0:2].T)
    group2=mat(np.array(groupgirl)[:,0:2].T)
    print("group1 :\n", group1)
    print("group2 :\n", group2)
    mean1 = compute_mean(group1)
    print("mean1 :\n", mean1.T)
    mean2 = compute_mean(group2)
    print("mean2 :\n", mean2)
    s_in1 = compute_withinclass_scatter(group1, mean1)
    print("s_in1 :\n", s_in1)
    s_in2 = compute_withinclass_scatter(group2, mean2)
    print("s_in2 :\n", s_in2)
    # 求总类内离散度矩阵
    s = s_in1 + s_in2
    print("s :\n", s)
    # 求s的逆矩阵
    s_t = s.I
    print("s_t :\n", s_t)
    # 求解权向量
    w = dot(s_t, mean1 - mean2)
    print("w :\n", w)
    return w,mean1,mean2,group1,group2


def predict(data, w, mean1, mean2):
    test1 = mat(data)
    g = dot(w.T, test1.T - 0.5 * (mean1 + mean2))
    return g

groupboy,l=getdata('boy.txt')
groupgirl,ls =getdata('girl.txt')
print(groupboy)
w,mean1,mean2,group1,group2= train(groupboy,groupgirl)
drawcompare(w,mean1,mean2,group1,group2,groupboy,groupgirl)

#-------------------------------------------------------
testboy,labels=getdata('boy82.txt')#女生标记为0，男生标记为1
group3=mat(np.array(testboy)[:,0:2].T)
print(shape(group3))
draw(w,mean1,mean2,group3,group2)
count=0
print('1:',(predict(group3[:,1].T,w,mean1,mean2)))
for i in range(shape(group3)[1]):
    if (predict(group3[:,i].T,w,mean1,mean2))[0,0]<=0:#这里要注意和测试数据对应
        count = count+1

print(count/shape(group3)[1])
# def liuyi(data1,data2,labels1,labels2):
#     alldata=data1+data2
#     alllabels=labels1+labels2
#     for i in range(shape(data1)[0]+shape(data2)[0]):
#         test = alldata[i,:]
#         train = data[0:i,:]+data[i+1,:]
#         w, mean1, mean2, group1, group2 = train(data1, data2)







