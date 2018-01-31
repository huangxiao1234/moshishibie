from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LDA
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']#使可视化可以显示中文

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

    manslabels= [classnumber]*shape(mansls)[0]
    return mansls,manslabels

def pca_new(dataMat,n):
    '''

    :param dataMat: 待降维的数据集（已转成矩阵）
    :param n: 需降到的维度
    :return: 降维后的数据集，排序后的特征值，排序后的特征向量
    '''

    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigvals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(-eigvals)
    taken_eigVects=eigVects[eigValInd[:n],:]#取前N个特征向量
    dataMat = dot(dataMat, taken_eigVects.T)
    final_eigvals=eigvals[eigValInd]
    final_eigVects=eigVects[eigValInd,:]

    return dataMat,final_eigvals,final_eigVects


#读入数据
mansls,manslabels1=getdata('boy.txt',0)
girlsls,girlslabels1=getdata('girl.txt',1)
m=mansls.copy()
g=girlsls.copy()
ml=manslabels1.copy()
gl=girlslabels1.copy()
ml.extend(gl)
alllabels = ml
m.extend(g)
alldatas = mat(m)
#训练集降维
down_Mat,eigvals,eigVects=pca_new(alldatas,2)
eigVects[:2]=eigVects[:2]*300#为了看得见投影面和数据点

#可视化
ax=plt.subplot(223,projection='3d')
plt.title('均一化')
ax.scatter(array((mansls-mean(mansls))/sum(mansls))[:, 0], array((mansls-mean(mansls))/sum(mansls))[:, 1], array((mansls-mean(mansls))/sum(mansls))[:, 2], c='red')
ax.scatter(array((girlsls-mean(girlsls))/sum(girlsls))[:, 0], array((girlsls-mean(girlsls))/sum(girlsls))[:, 1], array((girlsls-mean(girlsls))/sum(girlsls))[:, 2], c='green')

ax=plt.subplot(221,projection='3d')
plt.title('原始数据')
ax.scatter(array(mansls)[:, 0], array(mansls)[:, 1], array(mansls)[:, 2], c='red')
ax.scatter(array(girlsls)[:, 0], array(girlsls)[:, 1], array(girlsls)[:, 2], c='green')

ax=plt.subplot(222,projection='3d')
plt.title('投影平面')
plt.xlim(150,200)
plt.ylim(40,90)

ax.scatter(array(mansls)[:, 0], array(mansls)[:, 1], array(mansls)[:, 2], c='red')
ax.scatter(array(girlsls)[:, 0], array(girlsls)[:, 1], array(girlsls)[:, 2], c='green')
ax.plot_trisurf([0,eigVects[:2][0,0],eigVects[:2][1,0]], [0,eigVects[:2][0,1],eigVects[:2][1,1]],[0,eigVects[:2][0,2],eigVects[:2][1,2]])

plt.subplot(224)
plt.title('投影之后')
plt.scatter(array(down_Mat)[:len(mansls), 0],array(down_Mat)[:len(mansls), 1], c='red')
plt.scatter(array(down_Mat)[len(mansls):, 0], array(down_Mat)[len(mansls):, 1], c='green')

plt.show()

#--------------降维后再使用LDA----------------------
#用降维的数据测试LDA
w,mean1,mean2,group1,group2= LDA.train(down_Mat[:len(mansls), :],down_Mat[len(mansls):, :])
#将测试集降维
testboy,labels=LDA.getdata('boy82.txt',0)
testboy = mat(testboy)
#训练集降维
test_Mat,test_eigvals,test_eigVects=pca_new(testboy,2)
group3=test_Mat.T
count=0
for i in range(shape(group3)[1]):
    if (LDA.predict(group3[:,i].T,w,mean1,mean2))[0,0]>=0:#这里要注意和测试数据对应
        count = count+1
print('降维后的准确率：',count/shape(group3)[1])#降维后有0.96准确率，而没降维只有0.84(用身高，体重)