from databaseUtils import Database

from databaseUtils import getIntLabels





import time
from datetime import timedelta
import math
import random
import numpy as np
from numpy.random import seed


dataset = Database("vggfaces.json")
#dataset = Database("flowers17.json")

dataset.createHDF5DatabaseGroups()

dataset.openDatabase()
trainset = dataset.getTrainset()
devset = dataset.getDevset()
testset = dataset.getTestset()

classNames = dataset.getClassNames()

print(classNames)

clsLabelDict = {}

labelClsDict = {}

# assign each class name a unique integer
lbl=0
for clsname in classNames:
    clsLabelDict[clsname]= lbl
    labelClsDict[lbl] = clsname
    lbl+=1
    
    
trainStrLabels = trainset["strLabels"][:, 0]

trainLabels= getIntLabels(clsLabelDict,trainStrLabels)

print("trainLabels.shape",trainLabels.shape)

numBatch = len(trainset["index"])
start = trainset["index"][0,0]
end  = trainset["index"][0,1]    

imgsBatch1 = trainset["images"][start:end,:]
print(imgsBatch1.shape)
print(devset["images"].shape)

print("number of classes", len(clsLabelDict))
print("numbBatch = ", numBatch , "  approx num of imgs in each batch = ", end-start)
#print(type(trainLabels))
#print(trainLabels.shape)



dataset.closeDatabase()


