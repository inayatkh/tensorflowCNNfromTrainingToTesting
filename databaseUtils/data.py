'''
Created on Sep 22, 2017

@author: inayat
'''
from .conf import Confcommentjson

import os
import glob

from sklearn.utils import shuffle
import math
import cv2
import numpy as np

from .batchIndexer import BatchIndexer

import sys

import h5py

class Database(object):
    '''
    classdocs
    '''


    def __init__(self, jsonFileName):
        '''
        Constructor, 
        Data set initialize input parameters from datasetname.json file
        '''
        conf = Confcommentjson(jsonFileName)
        
        self.trainingDatasetFloder = conf["TRAINING_DATASET"]
        self.datasetName = conf["DATASET_NAME"]
        self.imageExt = conf["IMAGE_EXT"]
        
        self.miniBatchSize = conf["MINI_BATCH_SIZE"]
        self.miniBatchOutPath = conf["MINI_BATCH_DATASET_OUT_PATH"]
        self.hdf5DatabaseFileName = conf["HDF5_DATABASE_FILE"]
        
        self.bHdf5datasetOpen = False
        
        self.trainDBname="trainset"
        self.devDBname="devset"
        self.testDBname="testset"
        
        
        self.validationSize = conf["VALIDATION_SIZE"]
        self.testSize = conf["TEST_SIZE"]
        self.imageSize = conf["IMAGE_SIZE"]
        self.numChannels = conf["NUM_CHANNELS"]
        
        
        # get class names from trainingDataset path
        self.classNames =  []
        self.classFolderPath = {} # dictionary which contain class name and folder paths
        self.classTrainDevTestSets = {}
        for folder in os.listdir(self.trainingDatasetFloder):
            folderPath = os.path.join(self.trainingDatasetFloder, folder)
            if os.path.isdir(folderPath):
                self.classNames.append(folder)
                
                # number of files in each folder
                files = glob.glob(folderPath + "/*" + self.imageExt )
                self.classFolderPath[folder] = [folderPath, files]
                
                
                files = shuffle(files)
                
                numDevImages = math.floor(len(files) * self.validationSize)
                numTestImages = math.floor(len(files) * self.testSize)
                numTrainImages = len(files) - numDevImages - numTestImages 
                
                print(numTrainImages,numDevImages,numTestImages)
                self.classTrainDevTestSets[folder] = [files[0:numTrainImages],
                                                      files[numTrainImages: numTrainImages + numDevImages],
                                                      files[numTrainImages + numDevImages: ]
                                                      
                                                      ]
                
        
        self.printInputParameters()
        
    def printInputParameters(self):
        '''
        
        '''
        
        print("[input params values] trainingDasetFolder ", self.trainingDatasetFloder)
        print("[input params values] datasetName ", self.datasetName)
        print("[input params values] imageExt ", self.imageExt)
        print("[input params values] miniBatchSize " , self.miniBatchSize)
        print("[input params values] miniBatchOutPath " , self.miniBatchOutPath)
        print("[input params values] hdf5DatabaseFileName " , self.hdf5DatabaseFileName)
        
        print("[input params values] validationSize ", self.validationSize)
        print("[input params values] testSize ", self.testSize)
        
        print("[input params values] imageSize ", self.imageSize)
        print("[input params values] numChannels ", self.numChannels)
        
        
        print(" ******** parameters calculated from the given input ****")
        print("[input params values] classNames ", self.classNames)
        print("[input params values] Num of classes ", len(self.classNames))
        
        
        print("[input params values] class folder paths are ")
        
        for name in self.classNames :
            print("   {} : {} : {}".format(name,self.classFolderPath[name][0],
                                           len(self.classFolderPath[name][1])))
            print("   train:{}, Dev:{}, Test:{}".format(len(self.classTrainDevTestSets[name][0]),
                                                        len(self.classTrainDevTestSets[name][1]),
                                                        len(self.classTrainDevTestSets[name][2])
                                                       ))
    def preProcessImg(self, imageFilePath):
        
           
        # Reading the image using OpenCV
        image = cv2.imread(imageFilePath)
        
        # Resizing the image to our desired size and 
        # this preprocessing must be exactly in the sameway
        # during training and predicting
        
        image = cv2.resize(image, (self.imageSize, self.imageSize), 
                           cv2.INTER_LINEAR)
        
        image = image.astype('float32')
        
        image = np.multiply(image , 1.0/ 255.0)
        
        
        
        #The input to the network is of shape [None image_size image_size num_channels].
        # Hence we reshape.
        
        image = image.reshape(1, self.imageSize, self.imageSize, self.numChannels)
        
        #image = image.reshape(1, self.imageSize * self.imageSize * self.numChannels)
        
        #print("preprocess image shape 2",image.shape)
        return image

    def createHDF5DatabaseGroups(self):
        
        #print(self.classTrainDevTestSets.values().shape)
        
        # open HDF5 database
        self.db = h5py.File(self.miniBatchOutPath +"/" +  self.hdf5DatabaseFileName, mode='w')
        
        self.bHdf5datasetOpen = True
        
        # Create groups
        #self.trainDBname="trainset"
        #self.devDBname="devset"
        #self.testDBname="testset"
        
        trainDbgroup = self.db.create_group(self.trainDBname)
        devDbgroup = self.db.create_group(self.devDBname)
        testDbgroup = self.db.create_group(self.testDBname)
        
        trainClassNames=[]
        trainImgFileNames =[]
        devClassNames=[]
        devImgFileNames =[]
        testClassNames=[]
        testImgFileNames =[]
        
        for clsName, fileNames in self.classTrainDevTestSets.items():
            
            _tmpClsNames = [clsName] * len(fileNames[0])
            trainClassNames.extend(_tmpClsNames)
            trainImgFileNames.extend(fileNames[0])
            
            _tmpClsNames = [clsName] * len(fileNames[1])
            devClassNames.extend(_tmpClsNames)
            devImgFileNames.extend(fileNames[1])
            
            _tmpClsNames = [clsName] * len(fileNames[2])
            
            testClassNames.extend(_tmpClsNames)
            testImgFileNames.extend(fileNames[2])
            
            
            
        trainClassNames, trainImgFileNames = shuffle(trainClassNames,trainImgFileNames)
        devClassNames, devImgFileNames = shuffle(devClassNames,devImgFileNames)
        testClassNames, testImgFileNames = shuffle(testClassNames,testImgFileNames)
        
        
        print("[info] Total Number of Training Images {}".format(len(trainImgFileNames)))
        print("[info] Total Number of Dev/Val Images {}".format(len(devImgFileNames)))
        print("[info] Total Number of Test Images {}".format(len(testImgFileNames)))
        
        #self.createDatabase("testSet.hdf5",testClassNames, testImgFileNames)
        
        totalNumTrainImages = len(trainImgFileNames)
        totalNumDevImages = len(devImgFileNames)
        totalNumTestImages = len(testImgFileNames)
        
        
        print("[info] creating training dataset ....")
        trainBatchIndexer =self._createDatabaseGroup(trainDbgroup, trainClassNames, trainImgFileNames,
                                                      totalNumTrainImages)
        print("[info] creating dev dataset ....")
        devBatchIndexer =self._createDatabaseGroup(devDbgroup, devClassNames, devImgFileNames,
                                                   totalNumDevImages)
        if self.testSize > 0:
            print("[info] creating test dataset ....")
            testBatchIndexer =self._createDatabaseGroup(testDbgroup, testClassNames, testImgFileNames,
                                                        totalNumTestImages)
              
            
        #### 
        ##  print hdf5 database groups and keys name
        ##
        ####
        print("[info]  Database groups and keys name .....")
        print("    Groups names  in HDF5 Database ")
        print("    ", [group for group in self.db.keys()])
        print("    key names for each group in HDF5 Database ")
        print("    ",[key for key in self.db["trainset"]])
        # close the database
        self.db.close()
        self.bHdf5datasetOpen = False
        '''
        for name in self.classNames:
            
            for trainFileName in self.classTrainDevTestSets[name][0]:
                
                #print(trainFileName)
                img = self.preProcessImg(trainFileName)
                print(img.shape)
        '''
    
    def _createDatabaseGroup(self, dbGroupObj, classNames, imgFileNames, estNumImages ):
        
        '''
        
        hdf5DasetPath, estNumImages=500, maxBufferSize=500,
                 bdResizeFactor=2, verbose=True):
        
        '''
        
        #print("[info:data.py:createDatabase] Creating database named: '{}' for `{}` number of images".format(dbName, len(imgFileNames)))
        
        #database= BatchIndexer(self.miniBatchOutPath +"/" + dbName,
        batchIndexer= BatchIndexer(dbGroupObj,
                               estNumImages=estNumImages,maxBufferSize= self.miniBatchSize * 2,
                               bdResizeFactor=2,  numImgINBatch=self.miniBatchSize, verbose=False)
         
        miniBatchClassNames=[]
        miniBatchImgFileNames=[]
        miniBatchImages = None
        batchCounter =0
        
        for i in range(0, len(imgFileNames)):
            
            img = self.preProcessImg(imgFileNames[i])
            #print(img.shape)
            
            miniBatchClassNames.append(classNames[i])
            miniBatchImgFileNames.append(imgFileNames[i])
            
                 
            
            if(miniBatchImages is None):
                miniBatchImages = img
            else:
                miniBatchImages=np.vstack([miniBatchImages, img])
            
            batchCounter+=1
            #if (i % self.miniBatchSize ) == 0  and i > 1:
            if batchCounter ==  self.miniBatchSize:
            
                # print(miniBatchImages.shape, len(miniBatchClassNames), len(miniBatchImgFileNames))
                
                batchIndexer.add(miniBatchImgFileNames, miniBatchClassNames, miniBatchImages)
                
                miniBatchClassNames=[]
                miniBatchImgFileNames=[]
                miniBatchImages = None
                batchCounter=0
        
        if  miniBatchImages is not None:
            batchIndexer.add(miniBatchImgFileNames, miniBatchClassNames, miniBatchImages)
            
       
        
        batchIndexer.finish()        
        
        return batchIndexer
    
    
    
    def getClassNames(self):
        
        
        if self.bHdf5datasetOpen:
            
            trainset = self.db["trainset"]
            
            classNames = trainset["strLabels"][:,0]
            
            classNames = np.unique(classNames)
            
            return classNames
            
        else:
            print("[info] dataset is not open")
          
    def getTrainset(self):
        
        if self.bHdf5datasetOpen:
            trainset = self.db["trainset"]
            #return  trainset["images"] , trainset["strLabels"][:,0]
            return trainset
        else:
            print("[info] dataset is not open")
    
    def getDevset(self):
        
        if self.bHdf5datasetOpen:
            devset = self.db["devset"]
            #return  trainset["images"] , trainset["strLabels"][:,0]
            return devset
        else:
            print("[info] dataset is not open")
     
    
    def getTestset(self):
        
        if self.bHdf5datasetOpen:
            testset = self.db["testset"]
            #return  trainset["images"] , trainset["strLabels"][:,0]
            return testset
        else:
            print("[info] dataset is not open")
    
    def openDatabase(self):
        
        if not self.bHdf5datasetOpen:
            print("[info] opening hdf5 datset")
            self.db = h5py.File(self.miniBatchOutPath +"/"+self.hdf5DatabaseFileName ,"r")
            self.bHdf5datasetOpen = True
            
                
    def closeDatabase(self):
        if self.bHdf5datasetOpen :
            print("[info] closing hdf5 datset")
            self.db.close()
            
            self.bHdf5datasetOpen = False       
        
            
        
        