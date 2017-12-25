from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=99999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat*redEigVects.T) +meanVals
    return lowDDataMat,reconMat

def replaceNanwithMean():
    datMat = loadDataSet('boy.txt')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

dataMat = replaceNanwithMean()
#print(dataMat)
meanVals = mean(dataMat,axis =0)
#print(meanVals)
meanRemoved = dataMat - meanVals
#print(meanRemoved)
covMat = cov(meanRemoved,rowvar=0)
#print(covMat)
eigvals,eigVects = linalg.eig(mat(covMat))
w=eigvals[:2]
print(dataMat.T)
print(eigvals)
print(dot(eigVects[0:2],dataMat.T))
g=dot(eigVects[0:2],dataMat.T)
plt.figure(1)
ax = plt.subplot(111)
ax.scatter(array(g)[0], array(g)[1], s=30, c='red', marker='s')
plt.show()