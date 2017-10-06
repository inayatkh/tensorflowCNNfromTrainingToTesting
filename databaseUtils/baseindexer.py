'''
Created on Sep 25, 2017

@author: inayat
'''
from __future__ import print_function
import numpy as np
import datetime

class BaseIndexer(object):
    '''
     This class  is a base class from Which batchIndexer class is derived
    '''


    def __init__(self, estNumImages=500,
                  maxBufferSize=500, bdResizeFactor=2,
                   numImgINBatch=32, verbose=True):
        '''
        Constructor
        # this method stores the following
        
        hdf5DatasetGroupObj : the group object in a  hdf5 database
        estNumImages : estimated number of images in the dataset
        maxBufferSize : max buffer size, idicates the maximum num of images in
                        memory prior to flushing to disk
        bdResizeFactor : the resize factor of the dataset
        
        numImgINBatch : number of images in a miniBatch
        verbose : the verbosity settings
        
        
        '''
        
        #self.hdf5DatasetGroupObj = hdf5DatasetGroupObj
        self.estNumImages = estNumImages
        self.maxBufferSize = maxBufferSize
        self.bdResizeFactor= bdResizeFactor
        
        self.verbose = verbose
        
        self.numImgINBatch = numImgINBatch
        
        # initialize the indexes dictionary
        '''
        hdf5 datasets can be viewed as gigantic NumPy arrays.
        in order to access a given row or insert data into hdf5 dataset, we
        need to know the index of the row we are accessing; just link numpy array.
        This idxs will store the current indexes into the respective datasets
        
        '''
        self.idxs = {}
        
    def _writeBuffers(self):
        '''
          This function take all available buffers and flush them to the disk
         
         as this is a root class so we don't have any concept on the 
         number of buffers that need to flushed
        
        '''
        pass
        
 
        
    def _writeBuffer(self, dataset, datasetName, buf, idxName):
        
        '''
         a sigular method that accepts a single buffer and writes
         it to the disk
         
         dataset : The hdf5 dataset object that we are writing the buffer to
         datasetName : the name of the hdf5 dataset (eg, internal, hierarchical path)
         buf: the buffer that will be flushed to the disk
         idxName: the name of the key into the idx dictionary. This variable will 
         allow us to access the current pointer into the hdf5 dataset and ensure
         our images are written in sequentional order without overalaping each other
        
        
        '''
        
        '''
         when our buffer of images reaches the maxBufferSize. then we need
         to flush the buffer to the hdf5 dataset and reset the buffer
        '''
        
        #print(dataset.shape,idxName, datasetName)
        # if buffer is a list then compute the ending index based on list length
        if type(buf) is list:
            end = self.idxs[idxName] + len(buf)
        
        
        # otherwise, assume that the buf is a numpy/scipy array
        # so cmpute the ending index based on the array shape
        
        else:
            end = self.idxs[idxName] + buf.shape[0]
        
        
        self._debug("baseindexer.py:_writeBuffer end={} datasetName={} idxName={}".format(end,datasetName,idxName))   
            
        # the end  index is simply the current index value plus the number of
        #entries in the buffer
        
        # check to see if the dataset needs to be resized
        if end > dataset.shape[0]:
            self._debug("...............triggering `{}` database resize".format(datasetName))
            self._resizeDataset(dataset, datasetName, baseSize=end)
            # Since HDF5 datasets are actually NumPy arrays, this means 
            # that they have a pre-defined dimension, both in terms of number 
            # of rows and number of columns
        
        
        #  writes the buf  to the HDF5 dataset using standard Python array 
        # slicing. The start of the slice is specified by self.idxs[idxName] 
        #, which is simply an integer pointing to the next open row in the 
        # dataset. The end of the slice is determined by our calculation of 
        # end  earlier in the _writeBuffer  dataset.    
        # dump the buffer buf to file
        
        #self._debug("writing `{}` records from {} to {}".format(datasetName,self.idxs[idxName], end))
        
       
        dataset[self.idxs[idxName]:end] = buf
        
        #leave resetting the buffers and incrementing the idxs 
        # value to the function that called _writeBuffer of the derived class BatchIndexer 
            
    def _resizeDataset(self, dataset, dbName, baseSize=0, finished=0):
        '''
        
          dataset : the hdf5 dataset object that we are resizing
          dbName : the name of the hdf5 dataset
          baseSize: the base size of the dataset is assumed to be the 
                    total number of rows in the database plus
                    the total number of entries in the buffer
        
        '''
        '''
         When resizing our HDF5 dataset, we need to be able to both increase 
         the size of the dataset (such as when we’re indexing images and 
         adding more features to the dataset) and compact the size of the 
         dataset (such as when the feature indexing process has completed, 
         and we need to reclaim unused space).
         
         Luckily, increasing and decreasing the size of a given dataset is
          handled by the same h5py  function — all we need to do is determine
           the new size (i.e. NumPy shape) of the dataset.
        '''
        
        # grab the original size of the dataset
        self._debug("baseindexer.py:_resizeDataset: resizing dataset {}".format(dbName))
        origSize = dataset.shape[0]
        
        # check to see if we are finished writing rows to the dataset,
        # and if so make the new size the current index
        
        if finished > 0:
            newSize = finished
            
        # otherwise, we are enlarging the dataset so caculated the new size
        # of the dataset
        else:
            newSize = baseSize * self.bdResizeFactor
            
        # determine the shape of the to be resized dataset
        shape = list(dataset.shape
                     )
        shape[0] = newSize
        
        # show old versus new size of the dataset
        dataset.resize(tuple(shape))
        self._debug("baseindexer.py:_resizeDataset: old size of `{}` : {:,}, new size {:,}".format(dbName,
                                                                    origSize,
                                                                    newSize))
        self._debug("baseindexer.py:_resizeDataset: dataset={} resized shape {}".format(dbName,dataset.shape))
        '''
          You would be correct in saying that we are allocating a lot of extra rows in the dataset; however, this operation is not wasteful since we will be compacting the dataset after the feature indexing process has finished. Furthermore, it’s likely that we’ll be performing expanding operations multiple times, whereas a compaction operation is guaranteed to only happen once. Since resizing operations tend to be expensive, it’s much more beneficial to over-allocate space during the feature vector insertion process.
        
        '''
        
    def _debug(self, msg, msgType="[INFO]"):
        '''
           the _debug  method, which can be used to (optionally) write debugging messages to our terminal.
        '''
        # check to see if the message should be printed
        if self.verbose:
            print("**** {} {} - {}".format(msgType, msg, datetime.datetime.now()))
            
    @staticmethod
    def featureStack(array, accum=None, stackMethod=np.vstack):
        '''
         Since buffers can be NumPy arrays, we should use NumPy/SciPy methods to stack and append them together. 
         This method simply accepts two arrays: array , which will be appended to accum , the accumulated array (i.e. the buffer).
         Given these two arrays, the rest of the featureStack  method handles stacking and appending these arrays based on the 
         supplied stackMethod .

          If you are unfamiliar with the concept of array stacking, 
          I advise reading the joining arrays section of the NumPy documentation.
        
        '''
        
        # if the accumulated array is None, initialize it
        if accum is None:
            accum = array
            
 
        # otherwise, stack the arrays
        else:
            #print("DDDDD",array.shape, accum.shape)
            accum = stackMethod([accum, array])
 
        # return the accumulated array
        return accum
               
        
        