import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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


def canshu(data):
    manmeans = np.mean(data,axis=0)
    height=manmeans[0]#身高均值
    weight = manmeans[1]#体重均值
    shoesize = manmeans[2]#鞋码均值

    m = data - manmeans#xi-均值

    mvar = np.var(data,axis=0)

    hvar,wvar,svar = mvar[0:3]#方差，男生对应正态分布的参数
    return height,weight,shoesize,hvar,wvar,svar

def zhengtaifenbu(x1,x,z):
    return stats.norm.pdf(x1,x,math.sqrt(z))

def byesclassfier(m1,m2,g1,g2,x,w1,w2):#求出x为w1的后验概率，w2的则为1-w1,m1,m2,g1,g2分别为男女对应正态分布参数，x为输入,w1,w2,分别为两类的先验概率
    p2 = zhengtaifenbu(x,m1,m2)#类条件概率p(x|w1)
    px2 = zhengtaifenbu(x,g1,g2)*w2
    px = px2+p2*w1#p(x)
    p = p2*w1/px#后验概率
    return p

def duoweiclassifier(x,mansls,girls,xianyan1,xianyan2):
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
    return multivariate_normal.pdf(x, mean=np.array(mu.T)[0], cov=thu.T)

def duoweicanshu(testdata):
    data1=np.mat(np.array(testdata)[:,0:2].T)
    dimens, nums = data1.shape[:2]
    mu =  np.mean(data1, axis=1)
    k=data1-mu
    s_in = 0
    for i in range(nums):
        x = k[:, i]
        s_in += np.dot(x, x.T)
    thu= s_in/nums
    return mu,thu

def ROC(prediction,test):
    TP=0#预测真为真
    FP=0#预测真为假
    FN=0#预测假为真
    TN=0#预测假为假

    for i in range(np.shape(test)[0]):
        print(TP,TN,FN,FP)
        if prediction[i]==1 and test[i]==1:
            TP=TP+1
        elif prediction[i]==0 and test[i]==0:
            TN=TN+1
        elif prediction[i]==1 and test[i]==0:
            FN=FN+1
        elif prediction[i]==0 and test[i]==1:
            FP=FP+1
    return TP/(TP+FN),FP/(FP+TN)


#----------一维正太-----------------------------
# mansls,manslabels = getdata('boy.txt')
# girls,girlslabels = getdata('girl.txt')
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
testdata,testlabels = getdata('boy82.txt')#如果要预测女孩，那么要将getdata里的labels设置成0
mansls,manslabels = getdata('boy.txt')
girls,girlslabels = getdata('girl.txt')
alldata=testdata+girls
alllabels=testlabels+[0]*np.shape(girls)[0]
x=[]
y=[]
print(alllabels)
for j in np.arange(0.1,1,0.1):
    xianyan1=j
    xianyan2=1-j
    testls = []
    for i in range(np.shape(alldata)[0]):
        prediction = duoweiclassifier(alldata[i][0:2],mansls,girls,0.5,0.5)
        if prediction>j:
            testls.append(1)
        else:
            testls.append(0)
    print(testls)
    r1,r2=ROC(testls,alllabels)
    x.append(r1)
    y.append(r2)
plt.plot(y,x)
plt.show()

