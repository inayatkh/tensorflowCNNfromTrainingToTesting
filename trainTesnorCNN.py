
import dataset





import time
from datetime import timedelta
import math
import random
import numpy as np
from numpy.random import seed


from multiClassCNN import ConvNNTF

from databaseUtils import Database

from databaseUtils import getIntLabels


seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

''' Here the things that we are going to do in this script:
1. Define your graph:
      a) Create Convolutional Layers.
      b) Create Fully-connected and Flatten Layers
      c) Create Other optimizer, loss function and others
2. Prepare your input dataset for training. 
3. Evaluating your own results. 
''' 

dataset = Database("vggfaces.json")
#dataset = Database("flowers17.json")

#dataset.createHDF5DatabaseGroups()

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

###

#print(TensorCNN._create_biases(2))

y= ConvNNTF._convert_to_one_hot(trainLabels, len(classNames))
print(y.shape)

numClasses=len(classNames)
print("numClasses =", numClasses)

numBatches= len(trainset["index"])
print(numBatches)
start = trainset["index"][numBatches-1,0]
end  = trainset["index"][numBatches-1,1] 


exampleMulticlassCNN = ConvNNTF();

#exampleMulticlassCNN.buildExpCNN(dataset.imageSize,
#                      dataset.numChannels,
#                      numClasses)

exampleMulticlassCNN.buildVggFaceSmall(dataset.imageSize,
                      dataset.numChannels,
                      numClasses)


cnnModelname="./model/" + dataset.datasetName

numIterations=400

#exampleMulticlassCNN.graphdisp()
     
exampleMulticlassCNN.train(trainset, devset, testset,clsLabelDict, cnnModelname, numIterations)


dataset.closeDatabase()


