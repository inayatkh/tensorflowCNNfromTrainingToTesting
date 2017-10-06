'''
Created on Sep 24, 2017

@author: inayat
'''

from .baseindexer import BaseIndexer
import numpy as np
import h5py

import sys

'''
Structure of the HDF5 database group

dbname.hdf5

strLabels:
    0: filename1, roses
    1: filename2, sunflowers
    ...
    ...
    
index:
    0: 0, 32
    1: 32, 65
    
    ...
    ...
    
images:
    0: 0,  size of of flatten image array
    1: 0, size of flatten image array 


strLabels has a shape of (N,2 ) N is the total no of images
index has a shape of (B,2) stores  two integers where B is the total number
      of batches. These integers are indexed into the images dataset for batch b
      
images  preprocessed images  has a  ( M, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)     
'''
class BatchIndexer(BaseIndexer):
    '''
    Data in HDF5 is stored hirarchically, similar to how file system stores data
    
     Data is first defined in groups, where a group is a container-like structure
      which can hold datasets and other groups. Once a group has been defined, a 
      dataset can be created within the group. A dataset can be thought of as a 
      multi-dimensional array of a homogenous data type.
      
      What makes h5py  so awesome is the ease of interaction with the data.
       We can store huge amounts of data in our HDF5 dataset and manipulate the 
       data using NumPy.
       
       When using HDF5 with h5py , you can think of your data as a gigantic 
       NumPy array that is too large to fit into main memory, but can still 
       be accessed and manipulated just the same.
       
       sudo apt-get install libhdf5-serial-dev
       pip install h5py
       
       
       HDF5 files are meant to store multiple groups and datasets. 
       A group is a container-like structure which can hold datasets 
       (and other groups). A dataset, however, is meant to store a
        multi-dimensional array of a homogenous data type — 
        essentially a NumPy matrix.
        
        However, unlike NumPy arrays, HDF5 datasets support additional 
        features such as compression, error detection, and chunked I/O. 
        All that said, from our point of view, we’ll be treating HDF5 
        datasets as large NumPy arrays with shape , size , and dtype  
        attributes
    '''


    def __init__(self, hdf5DatasetGroupObj, estNumImages=500, maxBufferSize=500,
                 bdResizeFactor=2,  numImgINBatch=32, verbose=True):
        '''
        Constructor
        
        input 
        #hdf5DasetPath is the path where our hdf5 database/ batches will be save
        hdf5DatasetGroupObj : the group object in a  hdf5 database
        
        estNumImages approximate number of images in the database ( so that we can roughly approx may be the num batches)
        maxBufferSize the maximum buffer size of images to be stored in memory
        '''
        
        super(BatchIndexer, self).__init__(estNumImages,
                                           maxBufferSize, bdResizeFactor,
                                           numImgINBatch,verbose )
        # open the hdf5 database for writing and initialize the 
        # the datasets in a group
        #self.db = h5py.File(hdf5DatasetPath, mode='w')
        self.db = hdf5DatasetGroupObj
        
        #self.testDBgroup = self.db.create_group("test")
        
        self.strLabelsDB = None
        self.indexDB = None
        self.imagesDB = None
        
       
        
        
        # initialize the buffers for the 3 datasets
        self.strLabelsBuffer = None
        self.indexBuffer = []
        self.imagesBuffer = None
        
        # initialize the total number of images in the buffer along
        # with index dictionary
        self.totalImages = 0
        self.idxs = {"batchIndex" : 0, "imagesIdx": 0}
        '''
         the index dictionary idxs contains two entries
         batchIndex : an integer representing the current row (ie the next empty row)
                 in the batch index datasets
         imagesIdx : an integer representing the next empty row in bothe
                     the images and strLabels datasets
         
              Again, when you think of these two values as indexes into a NumPy array ,they make a lot more sense. Since all three image_ids , index , and features  datasets are empty, the next empty rows in the respective arrays are 0. It’s important to note that both of these values will be incremented as the buffers are filled and flushed to disk.
         
         '''
        
        
        
    def addMiniBatch(self):
        pass
    
    def add(self, miniBatchImageFileNames, miniBatchClassNames, miniBatchImages):
        
        # compute the starting and ending index for the images db lookup
        
        self._debug("[info:batchindexer.py:add] adding images to dataset , {} image files and classnames , batchsize {}"
              .format(len(miniBatchImageFileNames), miniBatchImages.shape))
        
        
        start = self.idxs["imagesIdx"] + self.totalImages
        end = start + len(miniBatchImageFileNames)
        
       
        
        
        # update the image ids_classNamesBuffer index buffer,
        # followed by incrementing the image count
        
        self.strLabelsBuffer =BaseIndexer.featureStack(np.array(list(zip(miniBatchClassNames, miniBatchImageFileNames))),
                                                                  self.strLabelsBuffer
                                                     )
        
        #self.strLabelsBuffer = BaseIndexer.featureStack(np.array(list(zip(miniBatchClassNames, miniBatchImageFileNames)))
        #                                                           , self.strLabelsBuffer)
        
        #print(type(miniBatchImages), miniBatchImages.shape)
        
        self.imagesBuffer = BaseIndexer.featureStack(miniBatchImages,
                                                     self.imagesBuffer
                                                      )
        
        self.indexBuffer.append((start,end))
        
        self.totalImages +=len(miniBatchClassNames)
        
        
        self._debug("[info:batchindexe.py:add] added images to dataset , {} image files and classnames , batchsize {}"
              .format(len(self.strLabelsBuffer), self.imagesBuffer.shape))
        
        self._debug("start={}  end={}  total images={} maxBufferSize={}".format(start,end, self.totalImages, self.maxBufferSize))
        
        if(self.totalImages >= self.maxBufferSize):
            # if the database has not been created yet then create it
            if None in (self.strLabelsDB, self.indexDB, self.imagesDB):
                self._debug("initial buffer full")
                self._createDatasets()
                
            self._writeBuffers()
                
                                
                               
        
       
        
        
    def _createDatasets(self):
        
       
 
        # grab the images batch size
        #
        #fvectorSize = self.imagesBuffer.shape[1]
        batchImageDBSize = (self.estNumImages, 
                          self.imagesBuffer.shape[1],
                          self.imagesBuffer.shape[2],
                          self.imagesBuffer.shape[3])
        
        
        # batchindex dataset size
        
        approxNumBatches = int( self.estNumImages / self.numImgINBatch)
 
        # initialize the datasets
        self._debug("[batchindexer.py:_createDatasets] creating hdf5  datasets batchImageDBSize {}".format(batchImageDBSize))
        
        
        self.strLabelsDB = self.db.create_dataset("strLabels", 
                                                             (self.estNumImages,2),
                                                             maxshape=(None,2),
                                                             dtype=h5py.special_dtype(vlen=str))
        
        #self.strLabelsDB = self.db.create_dataset("strLabels", 
        #                                                     (self.estNumImages,2),
        #                                                     maxshape=(None,2),
        #                                                     dtype=h5py.special_dtype(vlen=str))
        
        
        
        
        
        self.indexDB = self.db.create_dataset("index", (self.estNumImages, 2),
            maxshape=(None, 2), dtype="int")
        
        #self.featuresDB = self.db.create_dataset("features",
        #    (approxFeatures, fvectorSize), maxshape=(None, fvectorSize),
        #    dtype="float")
        
        self.imagesDB = self.db.create_dataset("images",
            batchImageDBSize,
            maxshape = (None, self.imagesBuffer.shape[1],
                          self.imagesBuffer.shape[2],
                          self.imagesBuffer.shape[3]),
                          dtype="float") 
        
        
        '''
         However, a very important attribute to pay attention to is the maxshape  parameter. As the name suggests, this value controls the maximum size of the HDF5 dataset. If no maxshape  is defined, the maximum number of rows and columns is assumed to be the shape  of the matrix. More importantly, if maxshape  is not defined, you will not be able to resize your dataset later! Instead, you can specify a value of None  along any dimension that you want to make resizable.
         
         Since we do not know the number of images we’ll be storing in image_ids , we’ll make the total number of rows None , indicating that we’ll be able to resize our image_ids  dataset and store as many image IDs as we want.
         
         Finally, we specify the dtype  of the imageIDDB  to be unicode  since we’ll be storing image filenames as strings.
         
         The indexDB  dataset is defined in a similar way, only using integers as the dtype  and a shape of (self.estNumImages, 2) , where the two integers will be our lookups into the features  dataset. We’ll also specify None  for the maxshape  rows, indicating we can add as many rows as we want (but the number of columns will stay fixed at 2).
         
         Finally, we define the featuresDB , naming it features  and giving it an initial shape of (approxFeatures, fvectorSize) . This dataset will be a floating point data type since RootSIFT features are floats. Finally, we’ll allow this dataset to have an unlimited number of rows as well.
         
         
         
        '''
        
        
    def _isAnyBufferEmpty(self):
        
        '''
         return True if any of the buffers are empty otherwise false
         
         
        '''
        if np.any(self.strLabelsBuffer == None):
            #print("none check")
            return True
        else:
            return False
            
    def _writeBuffers(self):
        # write the buffers to disk
        self._debug("batchindex.py:_writeBuffers writing Buffers ....")
        
        
        self._writeBuffer(self.strLabelsDB, 
                          "strLabels", 
                          self.strLabelsBuffer,
                          "imagesIdx")
        
        self._writeBuffer(self.indexDB,
                          "index", 
                          self.indexBuffer, 
                          "batchIndex")
        
        self._writeBuffer(self.imagesDB, "images", 
                          self.imagesBuffer,
                          "imagesIdx")
 
        # increment the indexes
        self.idxs["batchIndex"] += len(self.indexBuffer) #$self.strLabelsBuffer) 
        
        self.idxs["imagesIdx"] += self.totalImages
 
        
       
        self._debug("batchindexer.py:_writeBuffers  self.idxs[\"batchIndex\"]= {},  self.idxs[\"imagesIdx\"] ={}".format(
            self.idxs["batchIndex"], self.idxs["imagesIdx"]))
        
        
        # reset the buffers and feature counts
        self.strLabelsBuffer = None
        self.indexBuffer = []
        
        self.imagesBuffer = None
        self.totalImages = 0
        
    
    def finish(self):
        # if the databases have not been initialized, then the original
        # buffers were never filled up
        if None in (self.strLabelsDB, self.indexDB,
                     self.imagesDB):
            self._debug("batchindexer.py:finish: minimum init buffer not reached", msgType="[WARN]")
            self._createDatasets()
 
        # write any unempty buffers to file
        self._debug("batchindexer.py:finish: writing un-empty buffers...", "[INFO]")
        if not self._isAnyBufferEmpty():
            self._writeBuffers()
 
        # compact datasets
        self._debug("batchindexer.py:finish: compacting datasets...")
        
        
        self._resizeDataset(self.indexDB, 
                            "index",
                             finished=self.idxs["batchIndex"])
        
        
        self._resizeDataset(self.imagesDB, 
                            "images",
                             finished=self.idxs["imagesIdx"])
        
        self._resizeDataset(self.strLabelsDB,
                             "strLabels",
                             finished=self.idxs["imagesIdx"])
 
        # close the database
        #self.db.close()