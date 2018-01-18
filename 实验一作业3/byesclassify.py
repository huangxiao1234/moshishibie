import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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


def canshu(data):
    '''
    用于获得贝叶斯分类所需要的参数
    :param data: 原始数据列表
    :return: 身高、体重、鞋码的均值和方差
    '''
    manmeans = np.mean(data,axis=0)
    height=manmeans[0]#身高均值
    weight = manmeans[1]#体重均值
    shoesize = manmeans[2]#鞋码均值
    m = data - manmeans#xi-均值
    mvar = np.var(data,axis=0)
    hvar,wvar,svar = mvar[0:3]#方差，男生对应正态分布的参数
    return height,weight,shoesize,hvar,wvar,svar

def zhengtaifenbu(x1,mean,var):
    '''
    一维正态分布函数
    :param x1: X值
    :param x: 均值参数
    :param z: 方差参数
    :return:x对应的Y
    '''
    return stats.norm.pdf(x1,mean,math.sqrt(var))

def byesclassfier(manmean,manvar,girlmean,girlvar,x,w1xianyan,w2xianyan):
    '''
    一维正态分布分类
    求出x为w1的后验概率，w2的则为1-w1
    贝叶斯公式的实现
    :param manmean:男生样本均值
    :param manvar:男生样本方差
    :param girlmean:女生
    :param girlvar:女生方差
    :param x: 要预测的值
    :param w1xianyan: w1的先验概率
    :param w2xianyan: w2的先验概率
    :return:
    '''
    p2 = zhengtaifenbu(x,manmean,manvar)#类条件概率p(x|w1)
    px2 = zhengtaifenbu(x,girlmean,girlvar)*w2xianyan
    px = px2+p2*w1xianyan#p(x)
    p = p2*w1xianyan/px#后验概率
    return p

def duoweiclassifier(x,mansls,girls,xianyan1,xianyan2):
    '''
    多维正态分布分类
    :param x: 输入向量
    :param mansls: 男生数据样本
    :param girls: 女生数据样本
    :param xianyan1: w1先验概率
    :param xianyan2: w2先验概率
    :return:
    '''
    mu,thu=duoweicanshu(mansls)#通过函数求得两个变量值
    mu1,thu1=duoweicanshu(girls)
    y =duoweizhengtai(x,mu,thu)#p(x|w1)
    y1 = duoweizhengtai(x,mu1,thu1)#p(x|w2)
    px = y*xianyan1+y1*xianyan2#p(x)
    p = y*xianyan1/px#后验概率
    return p

def compare(p1,p2):
    if p1>=p2:
        return True
    else:
        return False

def accurcy(predict,test):
    count = 0
    n = np.shape(test)[0]
    for i in range(n) :
        if predict[i]==test[i]:
            count = count+1
    return count/n

def duoweizhengtai(x,mu,thu):
    '''
    多维正态分布
    :param x: 输入向量
    :param mu: 均值向量
    :param thu: 方差向量
    :return: X对应的正态分布值
    '''
    return multivariate_normal.pdf(x, mean=np.array(mu.T)[0], cov=thu.T)

def duoweicanshu(testdata):
    '''
    求多维正太分布需要的参数
    :param testdata: 数据集
    :return: 均值向量，方差向量
    '''
    data1=np.mat(np.array(testdata)[:,0:2].T)#这里选的是样本里面前两个特征值，若要修改则手动将[:,0:2]变成[:,1:3]
    dimens, nums = data1.shape[:2]
    mu =  np.mean(data1, axis=1)
    k=data1-mu
    s_in = 0
    for i in range(nums):
        x = k[:, i]
        s_in += np.dot(x, x.T)
    thu= s_in/nums
    return mu,thu

def parzen(alldata,x,h):
    '''
    parzen窗的实现，用的方窗

    :param alldata:所有数据集
    :param x: 窗的距离
    :param h: 窗的高度
    :return:
    '''
    n=alldata.shape[0]
    print(n)
    a=[]
    b=0
    print(alldata[0])
    for j in range(n):#遍历每个样本
        if abs((alldata[j]-x)/h) <= 0.5:#方窗
            q=1
        else:
            q=0
        b=q+b
    return b/n

def ROC(prediction,test,trueclass,flaseclass):
    '''

    :param prediction: 预测标志
    :param test: 真实标志
    :param trueclass: 视为真的类别值
    :param flaseclass: 视为假的类别值
    :return:
    '''
    #这里写的字母表示意义与教材的不一致，因此后面计算真阳率时不能用书上的公式
    TP=0#预测真为真
    FN=0#预测真为假
    FP=0#预测假为真
    TN=0#预测假为假

    for i in range(np.shape(test)[0]):
        print(TP,TN,FN,FP)
        if prediction[i]==trueclass and test[i]==trueclass:
            TP=TP+1
        elif prediction[i]==flaseclass and test[i]==flaseclass:
            TN=TN+1
        elif prediction[i]==trueclass and test[i]==flaseclass:
            FP=FP+1
        elif prediction[i]==flaseclass and test[i]==trueclass:
            FN=FN+1

    if (FP+TN)==0:#因为样本选取可能没有假例，此时预测假为假的TN值永远为零，当FP也为零时，分母为零，报错，这样做为了防止报错，因此最好不要用只有一类的样本集
        return TP/(TP+FN),0
    return TP/(TP+FN),FP / (FP + TN)


#----------一维正太-----------------------------
# mansls,manslabels = getdata('boy.txt'，0)
# girls,girlslabels = getdata('girl.txt'，1)
# height1,weight1,shoesize1,hvar1,wvar1,svar1 = canshu(mansls)
# height,weight,shoesize,hvar,wvar,svar = canshu(girls)
# xianyan1=0.8
# xianyan2=0.2
# #贝叶斯求后验概率公式中可以看到，第一类先验概率增大，后验概率也会增大，准确率也会上升
# p1=byesclassfier(height1,hvar1,height,hvar,180,xianyan1,xianyan2)
# #p2=byesclassfier(height,hvar,height1,hvar1,180,xianyan1,xianyan2)
# # print(compare(p1,p2))
# print('p1:',p1)
#
#
# #正太分布图
# # x= np.arange(100,250,0.1)
# # y  = zhengtaifenbu(x,height1,hvar1)
# # plt.plot(x,y)
# # plt.show()
#
# #预测数据
#testdata,testlabels = getdata('boy82.txt')#如果要预测女孩，那么要将getdata里的labels设置成0
# testls =[]
# for i in range(np.shape(testdata)[0]):
#     prediction = byesclassfier(height1, hvar1, height, hvar,testdata[i][0] , xianyan1, xianyan2)
#     if prediction>0.5:
#         testls.append(1)
#     else:
#         testls.append(0)
#
# print(accurcy(testls,testlabels))#准确率

#----------------------------------------------------------------

#----------------二维正态（互相独立）------------------------------
# mansls,manslabels = getdata('boy.txt')
# girls,girlslabels = getdata('girl.txt')
# xianyan1=0.5
# xianyan2=0.5
# x=[]
# y=[]
# z=[]
# for i in range(100,200):
#     for j in range(30,100):
#         prediction = duoweiclassifier([i,j], mansls, girls, xianyan1, xianyan2)
#         if 0.50 == round(prediction,2):
#             x.append(i)
#             y.append(j)
# plt.figure(1)
# group1=np.mat(np.array(mansls)[:,0:2].T)
# group2=np.mat(np.array(girls)[:,0:2].T)
# ax = plt.subplot(111)
# #ax.plot(range(0,200),range(0,200))
#
# ax.scatter(np.array(group1)[0], np.array(group1)[1], s=30, c='red', marker='s')  # 训练数据散点图
# ax.scatter(np.array(group2)[0], np.array(group2)[1], s=30, c='green')
# print(x)
# print(y)
# plt.plot(x,y)
# plt.show()

#-----------三维作图求交线--------------------------
#
# x=[]
# y=[]
# z=[]
# for i in range(100,200):
#     for j in range(30,100):
#         x.append(i)
#         y.append(j)
#         prediction = duoweiclassifier([i,j], mansls, girls, xianyan1, xianyan2)
#         z.append(prediction)
# print(z)
# fig = plt.figure()
# fig1 = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax1 = fig1.add_subplot(111, projection='3d')
# ax.plot_trisurf(x, y, z)
# ax.plot_trisurf(x, y, [0.5]*np.shape(x)[0])
# plt.show()
#-----------------------错误率作图------------------------
# x = []
# y = []
#
# for j in np.arange(0,1,0.2):
#     xianyan1=j
#     xianyan2=1-j
#     testdata,testlabels = getdata('boy82.txt')#如果要预测女孩，那么要将getdata里的labels设置成0
#     testls =[]
#     for i in range(np.shape(testdata)[0]):
#         prediction = duoweiclassifier(testdata[i][0:2],mansls,girls,xianyan1,xianyan2)
#         if prediction>0.5:
#             testls.append(1)
#         else:
#             testls.append(0)
#     x.append(j)
#     y.append(accurcy(testls,testlabels))
#
# plt.plot(x,y)
# plt.show()
#-----------------------------------------------------------------
if __name__ == '__main__':

    testdata,testlabels = getdata('boy82.txt',1)#男孩标1，女孩标0
    mansls,manslabels = getdata('boy.txt',1)
    girls,girlslabels = getdata('girl.txt',0)
    testdata.extend(girls)#为了使测试样本更全面，直接把训练集女孩跟boy82加在一起当测试集来用
    testlabels.extend([0]*np.shape(girls)[0])
    x=[]
    y=[]
    #预测测试集
    for j in np.arange(0,1,0.01):
        testls = []#预测结果labels
        for i in range(np.shape(testdata)[0]):
            prediction = duoweiclassifier(testdata[i][0:2],mansls,girls,0.5,0.5)
            if prediction>j:
                testls.append(1)
            else:
                testls.append(0)

        r1,r2=ROC(testls,testlabels,0,1)#把女孩视为真
        x.append(r1)
        y.append(r2)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(y, x)
    plt.show()

    #-------------------------parzen方窗测试-------------------------
    # mansls, manslabels = getdata('boy.txt')
    # x=[]
    # y=[]
    # alldata = np.sort(np.array(mansls)[:, 0])
    # for i in range ((np.array(mansls)[:,0]).shape[0]):
    #
    #
    #     y.append(parzen(alldata,alldata[i],10))
    #     x.append(alldata[i])
    #
    #
    # plt.plot(x,y)
    # plt.show()







