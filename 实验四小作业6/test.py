
import KNN as kn
import numpy as np

f  = open('boy.txt')
lines  = f.readlines()
mansls= []
for line in lines:
    a=line.split()
    a= [float(i) for i in a ]
    mansls.append(a)
# labels = np.zeros(np.shape(ls)[0])
manslabels= [1]*np.shape(mansls)[0]

f1  = open('girl.txt')
lines1  = f1.readlines()
girlsls= []
for line1 in lines1:

    b=line1.split()
    b=[float(i) for i in b]
    girlsls.append(b)
# labels = np.zeros(np.shape(ls)[0])
girlslabels= [0]*np.shape(mansls)[0]


manslabels.extend(girlslabels)
alllabels = manslabels
mansls.extend(girlsls)
alldatas = np.array(mansls)

for j in range(1,10):
    print(kn.classify0([152.,49.,36.],alldatas,alllabels,j))