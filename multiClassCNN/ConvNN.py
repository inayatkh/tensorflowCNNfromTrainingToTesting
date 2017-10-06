'''
Created on Oct 2, 2017

@author: inayat
'''


# for shutting down the tensor flow warinings 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf
import numpy as np

from databaseUtils  import getIntLabels

#import gflags
#gflags.ADOPT_module_key_flags(tf)
#gflags.FLAGS(['-minloglevel', '3'])


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)



class ConvNNTF(object):
    '''
    This a general class for building 
    a tesnorflow based CNN
    '''
    
    _verbose = True
    
    _alpha_leakyRELU = 0.01
    
    _learning_rate = 0.0001


    def __init__(self):
        '''
        Constructor
        '''
        
    @staticmethod
    def _debug(msg, msgType="[INFO]"):
        '''
           the _debug  method, which can be used to (optionally) write debugging messages to our terminal.
        '''
        # check to see if the message should be printed
        if ConvNNTF._verbose:
            print("**** {} {} ".format(msgType, msg))
            
            
    
    @staticmethod
    def _create_weights(shape):
        #weights can be initialized using different methods
        # here we use normal distribution (with mean zero and small variation)
        
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))#, dtype=tf.float32))
    
    @staticmethod
    def _create_biases(size):
        
        return tf.Variable(tf.constant(0.05, shape=[size]))
    
    @staticmethod
    def _create_pad_mat(padSize):
        
        padMat = np.array([[0,0], [padSize, padSize],[padSize, padSize],[0,0]])
        return padMat
    
    
    @staticmethod
    def _create_pooling_layer(idx,inputs, size=2,stride=2):
        
               
        
        
        ConvNNTF._debug("    Layer  {} : Type = MAX Pool, Size = {} x {}, Stride = {}".format(idx,size, size, stride))
                            
        
        maxPoolLayer = tf.nn.max_pool(value=inputs, ksize=[1, size, size, 1],
                                      strides=[1, stride, stride, 1],
                                      padding='SAME',
                                      name=str(idx)+'_pool')
         
        layerShape = inputs.get_shape()                  
        ConvNNTF._debug("                         input  shape ={}".format(layerShape))
        layerShape = maxPoolLayer.get_shape() 
        ConvNNTF._debug("                         output shape ={}".format(layerShape))                 
                            
        return maxPoolLayer
        
        
    @staticmethod
    def _create_FC_layer(idx,inputs, numInputs,
                         numOutputs,use_leaky_relu=True, linear=False
                         ):
        
        Weights = ConvNNTF._create_weights(shape=[numInputs, numOutputs])
        Biases = ConvNNTF._create_biases(numOutputs)
        
        ConvNNTF._debug("    Layer  {} : Type = Full, Hidden/outputs = {}, Input dimension = {},Activation = {}".format(idx,numOutputs,
          
                                                                                                                        numInputs, 1-int(use_leaky_relu)) )
        layerShape = inputs.get_shape()                  
        ConvNNTF._debug("                         input  shape ={}".format(layerShape))
        
        
        if linear:
            
            fcLayer =  tf.add(tf.matmul(inputs,Weights),Biases,name=str(idx)+'_fc_linear')
            
            layerShape = fcLayer.get_shape() 
            ConvNNTF._debug("                         output shape ={}".format(layerShape))
            return fcLayer
        
        
        if use_leaky_relu :
            
            ipLayer = tf.add(tf.matmul(inputs,Weights),Biases)
            
            #fcLayer =  tf.maximum(ConvNNTF._alpha_leakyRELU * ipLayer ,ipLayer ,name=str(idx)+'_fc')
            fcLayer = ConvNNTF._create_RELU_layer(idx, ipLayer)
            
            layerShape = fcLayer.get_shape() 
            ConvNNTF._debug("                         output shape ={}".format(layerShape))
            
            
            return fcLayer
            
            
        else:
            
            ipLayer = tf.add(tf.matmul(inputs,Weights),Biases)
            
            #fcLayer =  tf.nn.relu(ipLayer,name=str(idx)+'_fc')
            fcLayer = ConvNNTF._create_RELU_layer(idx, ipLayer)
            
            layerShape = fcLayer.get_shape() 
            ConvNNTF._debug("                         output shape ={}".format(layerShape))
            
            return fcLayer
            
        
            
    
    @staticmethod
    def _create_flatten_layer(layer):
        # shape of the layer is [batchSize, imgSize, imgSize, numChannels]
        # but let get it from the previous layer
        
        layerShape = layer.get_shape()
        #print("_create_flatten_layer:layerShape", layerShape)
        
        # number of features will be imgHeight x imgHeight x numChannels
        
        numFeatures = layerShape[1:4].num_elements()
        
        # flatten the layer
        layer = tf.reshape(layer,[-1,numFeatures])
        
        return layer
    
    @staticmethod 
    def _create_RELU_layer(idx, inputs):
        
        ConvNNTF._debug("    Layer  {} : Type = RELU_layer".format(idx))
        
        outputs = tf.nn.relu(inputs, name=str(idx)+'_RELU')
        
        layerShape = inputs.get_shape()                  
        ConvNNTF._debug("                         input  shape ={}".format(layerShape))
        layerShape = outputs.get_shape() 
        ConvNNTF._debug("                         output shape ={}".format(layerShape))
        
        return outputs
    
    @staticmethod 
    def _create_LEAKY_RELU_layer(idx, inputs):
        
        ConvNNTF._debug("    Layer  {} : Type = Leaky_ RELU_layer".format(idx))
        
        outputs = tf.maximum(ConvNNTF._alpha_leakyRELU * inputs , inputs ,name=str(idx)+'_LEAKY_RELU')
    
        layerShape = inputs.get_shape()                  
        ConvNNTF._debug("                         input  shape ={}".format(layerShape))
        layerShape = outputs.get_shape() 
        ConvNNTF._debug("                         output shape ={}".format(layerShape))
        
        return outputs
        
    
    @staticmethod
    def _create_conv2d_layer(idx,inputs, numInputChannels,
                             convFilterSize,
                             numFilters,stride=1,padSize =1):
        
        
        Weights = ConvNNTF._create_weights(shape=[convFilterSize, convFilterSize, numInputChannels, numFilters])
        Biases = ConvNNTF._create_biases(numFilters)
        
        padSize = int(convFilterSize /2)
        
        
        padMat = ConvNNTF._create_pad_mat(padSize)
        
        
        inputsPad = tf.pad(inputs,padMat)
        
        
        convLayer = tf.nn.conv2d(input=inputsPad,
                                  filter=Weights,
                                  strides=[1, stride, stride, 1],
                                  padding='VALID',
                                  name=str(idx)+'_conv2d')
        '''
        
        
        convLayer = tf.nn.conv2d(input=inputs,
                                  filter=Weights,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name=str(idx)+'_conv2d')
        
        '''
        
        convLayerBiased = tf.add(convLayer,Biases,name=str(idx)+'_conv2d_biased')
        
        ConvNNTF._debug("    Layer  {} : Type = Conv, Size = {} x {}, Stride = {}, Filters = {}, Input channels = {}".format(idx, 
                                                                                                                                  convFilterSize,
                                                                                                                                  convFilterSize,
                                                                                                                                  stride,numFilters,
                                                                                                                                  numInputChannels))
        #pool_layer = ConvNNTF._create_pooling_layer(idx=idx+1, inputs=convLayerBiased,
        #                                          size=2, stride=2)
            
        #return tf.maximum(TensorCNN.alpha*convBiased,
        #                  convBiased,name=str(idx)+'_leaky_relu')
        '''
        convBiased = TensorCNN._pooling_layer(idx, convBiased,2, 2)
        '''
            
        
        #return tf.nn.relu(pool_layer)
        
        layerShape = inputs.get_shape()                  
        ConvNNTF._debug("                         input  shape ={}".format(layerShape))
        layerShape = convLayerBiased.get_shape() 
        ConvNNTF._debug("                         output shape ={}".format(layerShape))
        
        return convLayerBiased
    
    
    
    def buildExpCNN(self, imgSize, numChannels, numClasses ):
        
        
        self.logsPathTrain = "./tensorboardlogs/logTrain"
        self.logsPathVal = "./tensorboardlogs/logVal"
        
        ConvNNTF._debug("............ Building Deep Conv Neural Network .......")
        
        self.session = tf.Session()
        
        self.x =  tf.placeholder(tf.float32, shape=[None, imgSize,imgSize,
                                                    numChannels], name='x')
        
        self.y_true = tf.placeholder(tf.float32, shape=[None, numClasses],
                                     name='y_true')
        
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)
        
        
        self.conv_1 = self._create_conv2d_layer(idx=1,inputs=self.x,
                                                numInputChannels=numChannels,
                                                convFilterSize=3,
                                                numFilters=32,
                                                stride=1)
                                                #padSize=2) #fff
        '''
         conv layer accepts an input of size W1 x H1 X D1
                    produces a volume of size W2= (W1 - F + 2P) / S + 1 
                                            ==> (150 - 3 + 2*1)/1 + 1 = 150
                                            H2 = (H1 - F + 2*P)/ S +1
                                                ===> 150
                                                
                                            D2 = K
                    hyperparameters are numbFilters , K
                                     their spatial extent or filter size ie convFilterSize
                                     the sride S
                                     the amount of zero padding ie padSize    
                    
                 
        '''
        
        self.max_pool_2 = self._create_pooling_layer(idx=2, inputs=self.conv_1, size=2, stride=2)
        
        '''
         maxPool layer accepts an input of size W1 x H1 X D1
                    produces a volume of size W2= (W1 - F) / S + 1 
                                            ==> (150 - 2 )/2 + 1 = 75
                                            H2 = (H1 - F)/ S +1
                                                ===> 75
                                            D2 = D1
                    hyperparameters  their spatial extent or filter size ie size or F
                                     the sride S
                                    
                                    
                    
                 
        '''
        
        #self.leakRelu_3 = self._create_LEAKY_RELU_layer(idx=3, inputs=self.max_pool_2)
        self.Relu_3 = self._create_RELU_layer(idx=3, inputs=self.max_pool_2)
        
        
        
        #self.conv_4 = self._create_conv2d_layer(idx=4,inputs=self.leakRelu_3,
        self.conv_4 = self._create_conv2d_layer(idx=4,inputs=self.Relu_3,
                                                numInputChannels=32,
                                                convFilterSize=3,
                                                numFilters=32,
                                                stride=1)
        
        self.max_pool_5 = self._create_pooling_layer(idx=5, inputs=self.conv_4, size=2, stride=2)
        
        #self.leakRelu_6 = self._create_LEAKY_RELU_layer(idx=6, inputs=self.max_pool_5)
        self.Relu_6 = self._create_RELU_layer(idx=6, inputs=self.max_pool_5)
        
        
        #self.conv_7 = self._create_conv2d_layer(idx=4,inputs=self.leakRelu_6,
        self.conv_7 = self._create_conv2d_layer(idx=4,inputs=self.Relu_6,
                                                numInputChannels=32,
                                                convFilterSize=3,
                                                numFilters=64,
                                                stride=1)
        
        self.max_pool_8 = self._create_pooling_layer(idx=5, inputs=self.conv_7, size=2, stride=2)
        
        #self.leakRelu_9 = self._create_LEAKY_RELU_layer(idx=9, inputs=self.max_pool_8)
        self.Relu_9 = self._create_RELU_layer(idx=9, inputs=self.max_pool_8)
        
        
        #self.flattenLayer_9 = self._create_flatten_layer(self.leakRelu_9)
        self.flattenLayer_9 = self._create_flatten_layer(self.Relu_9)
        
        self.fc_10 = self._create_FC_layer(idx=10, inputs=self.flattenLayer_9,
                                           numInputs=self.flattenLayer_9.get_shape()[1:4].num_elements(),
                                           numOutputs=64, use_leaky_relu=False, linear = False)
        #                                  numOutputs=64, use_leaky_relu=True, linear = False)  
        
        
        self.fc_11 = self._create_FC_layer(idx=11, inputs=self.fc_10,
                                           numInputs=64,
                                           numOutputs=numClasses, use_leaky_relu=False , linear= True)
        
        
        
        self.y_pred = tf.nn.softmax(self.fc_11,name='y_pred')
        
        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)
        
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)
        
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_11,
                                                    labels=self.y_true)
        
        self.cost = tf.reduce_mean(self.cross_entropy)
        
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)
        
        
        
        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 
                                               tf.float32))
        
        self.init = tf.global_variables_initializer()
        
        # add variable for tensorboard
        self.lossSummary = tf.summary.scalar("loss", self.cost)
        self.accuracySummary = tf.summary.scalar("acc", self.accuracy)
        # create two summary files inorder to visuallize them together
        self.summaryWriterTrain = tf.summary.FileWriter(self.logsPathTrain, graph=tf.get_default_graph())
        self.summaryWriterVal = tf.summary.FileWriter(self.logsPathVal, graph=tf.get_default_graph())
        
        
        # initailize all variables 
        
        self.session.run(self.init) 
        
        #self.session = tf.Session()
        
        self.saver = tf.train.Saver()
        
        self.totalIterations = 0
    
    
    def buildVggFaceSmall(self, imgSize, numChannels, numClasses ):
        
        
        self.logsPathTrain = "./tensorboardlogs/logTrain"
        self.logsPathVal = "./tensorboardlogs/logVal"
        
        ConvNNTF._debug("............ Building Deep Conv Neural Network .......")
        
        self.session = tf.Session()
        
        self.x =  tf.placeholder(tf.float32, shape=[None, imgSize,imgSize,
                                                    numChannels], name='x')
        
        self.y_true = tf.placeholder(tf.float32, shape=[None, numClasses],
                                     name='y_true')
        
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)
        
        
        self.conv_1 = self._create_conv2d_layer(idx=1,inputs=self.x,
                                                numInputChannels=numChannels,
                                                convFilterSize=3,
                                                numFilters=64,
                                                stride=1)
                                                #padSize=2) #fff
                                                
        
        
        self.Relu_1 = self._create_RELU_layer(idx=2, inputs=self.conv_1)
        
        self.conv_2 = self._create_conv2d_layer(idx=3,inputs=self.Relu_1,
                                                numInputChannels=64,
                                                convFilterSize=3,
                                                numFilters=64,
                                                stride=1)
        
        self.Relu_2 = self._create_RELU_layer(idx=4, inputs=self.conv_2)
        
        
        
        self.max_pool_1 = self._create_pooling_layer(idx=5, inputs=self.Relu_2, 
                                                     size=2, stride=2)
        
       
        
                
        
        
        #self.conv_4 = self._create_conv2d_layer(idx=4,inputs=self.leakRelu_3,
        self.conv_3 = self._create_conv2d_layer(idx=6,inputs=self.max_pool_1,
                                                numInputChannels=64,
                                                convFilterSize=3,
                                                numFilters=128,
                                                stride=1)
        
        self.Relu_3 = self._create_RELU_layer(idx=7, inputs=self.conv_3)
        
        self.conv_4 = self._create_conv2d_layer(idx=8,inputs=self.Relu_3,
                                                numInputChannels=128,
                                                convFilterSize=3,
                                                numFilters=256,
                                                stride=1)
        
        self.Relu_4 = self._create_RELU_layer(idx=9, inputs=self.conv_4)
        
        self.max_pool_2 = self._create_pooling_layer(idx=10, inputs=self.Relu_4, size=2, stride=2)
        
        
        self.conv_5 = self._create_conv2d_layer(idx=11,inputs=self.max_pool_2,
                                                numInputChannels=256,
                                                convFilterSize=3,
                                                numFilters=512,
                                                stride=1)
        
        self.Relu_5 = self._create_RELU_layer(idx=12, inputs=self.conv_5)
        
        self.max_pool_3 = self._create_pooling_layer(idx=13, inputs=self.Relu_5, size=2, stride=2)
        
        self.conv_6 = self._create_conv2d_layer(idx=13,inputs=self.max_pool_3,
                                                numInputChannels=512,
                                                convFilterSize=3,
                                                numFilters=128,
                                                stride=1)
        
        self.Relu_6 = self._create_RELU_layer(idx=14, inputs=self.conv_6)
        
        self.max_pool_4 = self._create_pooling_layer(idx=15, inputs=self.Relu_6, size=2, stride=2)
        
        
        #self.flattenLayer_9 = self._create_flatten_layer(self.leakRelu_9)
        self.flattenLayer_1 = self._create_flatten_layer(self.max_pool_4)
        
        self.fc_1 = self._create_FC_layer(idx=16, inputs=self.flattenLayer_1,
                                           numInputs=self.flattenLayer_1.get_shape()[1:4].num_elements(),
                                           numOutputs=6272, use_leaky_relu=False, linear = False)
        #                                  numOutputs=64, use_leaky_relu=True, linear = False)  
        
        
        self.fc_2 = self._create_FC_layer(idx=17, inputs=self.fc_1,
                                           numInputs=6272,
                                           numOutputs=numClasses, use_leaky_relu=False , linear= True)
        
        
        
        self.y_pred = tf.nn.softmax(self.fc_2,name='y_pred')
        
        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)
        
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)
        
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_2,
                                                    labels=self.y_true)
        
        self.cost = tf.reduce_mean(self.cross_entropy)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        
        
        
        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 
                                               tf.float32))
        
        self.init = tf.global_variables_initializer()
        
        # add variable for tensorboard
        self.lossSummary = tf.summary.scalar("loss", self.cost)
        self.accuracySummary = tf.summary.scalar("acc", self.accuracy)
        # create two summary files inorder to visuallize them together
        self.summaryWriterTrain = tf.summary.FileWriter(self.logsPathTrain, graph=tf.get_default_graph())
        self.summaryWriterVal = tf.summary.FileWriter(self.logsPathVal, graph=tf.get_default_graph())
        
        
        # initailize all variables 
        
        self.session.run(self.init) 
        
        #self.session = tf.Session()
        
        self.saver = tf.train.Saver()
        
        self.totalIterations = 0
        
        
    
    def show_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss):
        
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)
        
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
         
        self._debug(msg.format(epoch + 1, acc, val_acc, val_loss), "[PROGRESS]")
        
    @staticmethod
    def _convert_to_one_hot(Y, C):
        #Y = np.eye(C)[Y.reshape(-1)].T
        Y = np.eye(C)[Y.reshape(-1)]
        return Y
    
    
    def train(self, trainset, devset, testset,clsLabelDict, cnnModelname, numIterations=400):  
        
        # specify how often we want to print the accuracy and loss to terminal
        # and how often to the tensorboard summary
        
        skipPrint =10 # print every 10 iterations
        skipSummary = 1 # add summary every iteration
        
        num_examples = trainset["images"].shape[0]
        numBatches= len(trainset["index"])
        
        batch_size = trainset["index"][0,1] - trainset["index"][0,0]  
        batchCounter =0
        epoch = 0
        
        x_valid = devset["images"]
        valid_cls = devset["strLabels"][:,0]
        y_valid = getIntLabels(clsLabelDict,
                                    valid_cls) 
        y_valid = self._convert_to_one_hot(y_valid, len(clsLabelDict))
        
        feed_dict_val = {self.x: x_valid,
                             self.y_true: y_valid}
            
        for i in range(self.totalIterations,
                   self.totalIterations + numIterations):
            
            
            start = trainset["index"][batchCounter,0]
            end  = trainset["index"][batchCounter,1] 
            
            x_batch = trainset["images"][start:end,:]
            
            cls_batch = trainset["strLabels"][start:end, 0]
            y_true_batch = getIntLabels(clsLabelDict,
                                        cls_batch
                                         )
            
            y_true_batch = self._convert_to_one_hot(y_true_batch,
                                                     len(clsLabelDict))
            
            
            feed_dict_tr = {self.x: x_batch,
                            self.y_true: y_true_batch}
            
            #feed_dict_val = {self.x: x_valid,
            #                 self.y_true: y_valid}
            
            
            self.session.run(self.optimizer, feed_dict=feed_dict_tr)
            
            if i % skipPrint == 0: # print acc and loss to terminal
                
                train_acc =  self.session.run(self.accuracy, feed_dict= feed_dict_tr)
                train_loss = self.session.run(self.cost, feed_dict = feed_dict_tr)
                
                val_acc = self.session.run(self.accuracy, feed_dict= feed_dict_val)
                val_loss = self.session.run(self.cost, feed_dict = feed_dict_val)
                
                #print("{}, train ({}, {}), val({},loss{})".format(i, train_acc, train_loss, val_acc, val_loss))
                print(i, train_acc, train_loss, val_acc, val_loss)
                
            if i % skipSummary == 0 : # write summary to  file
                summary = self.session.run(self.lossSummary, feed_dict = feed_dict_tr)
                self.summaryWriterTrain.add_summary(summary, i)
                summary = self.session.run(self.accuracySummary, feed_dict = feed_dict_tr)
                self.summaryWriterTrain.add_summary(summary, i)
                
                summary = self.session.run(self.lossSummary, feed_dict = feed_dict_val)
                self.summaryWriterVal.add_summary(summary, i)
                summary = self.session.run(self.accuracySummary, feed_dict = feed_dict_val)
                self.summaryWriterVal.add_summary(summary, i)
                
            
            
                        
            if batchCounter == numBatches-1 :
                
                print(i, num_examples,batchCounter)
                #val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
                
                
                #epoch = int(i / int(num_examples/batch_size))
            
                #self.show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                
                
                self.saver.save(self.session, cnnModelname+str(epoch),global_step=i,
                                write_meta_graph=False)
                
                print("ephoch = {}".format(epoch))
                
                epoch+=1
                
                batchCounter = 0
                
            else:
                batchCounter+=1
                if batchCounter > numBatches-1:
                    batchCounter = 0
                    
                
                
                
        self.totalIterations += numIterations 
        self.session.close()
        
        
        
        
        
        
        
       
    
       
           
    
        
