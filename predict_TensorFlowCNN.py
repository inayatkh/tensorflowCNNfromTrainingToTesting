'''
Created on Oct 2, 2017

@author: inayat
'''
import numpy as np
import argparse
import imutils
import time
import cv2

# for shutting down the tensor flow warinings 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf

import sys

if __name__ == '__main__':
    
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image file")
    args = vars(ap.parse_args())
   
    imageSize = 150
    numChannels = 3
   
    # Reading the image using OpenCV
    image = cv2.imread(args["image"])
    if image is None:
        print("can not load image: ", args["image"])
        sys.exit()
    
    imageOrig = image.copy()
   
    # resize and preprocess the test image exactly in the same manner
    # as done in training of the network
   
    image = cv2.resize(image, (imageSize, imageSize),
                       cv2.INTER_LINEAR)
        
    image = image.astype('float32')
        
    image = np.multiply(image , 1.0 / 255.0)
   
    # The input to the network is of shape [None imageSize imageSize numChannels]. Hence we reshape
    image = image.reshape(1, imageSize, imageSize, numChannels)
   
    # now restore the trained model
    
    sess = tf.Session()
    
    # S-1 recreate the network graph
    datasetName = "flowers" # use the same name from flowers.json file
    cnnModelname="./model/" + datasetName + ".meta"
    saver = tf.train.import_meta_graph(cnnModelname)
    
    # S-2 load the network weights saved
    saver.restore(sess, tf.train.latest_checkpoint("./model/"))
    
    # access the default graph which we have restored
    graph = tf.get_default_graph()
    
    # in the trained network y_pred is the tensor that is the prediction of network
    y_pred = graph.get_tensor_by_name("y_pred:0")
    
    # feed the test image to the input placeholder
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images  = np.zeros((1,2))
    
    ## creat feed_dic which is required to be feed inorder to calculat y_pred
    feed_dic_testing = {x:image, y_true:y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dic_testing)
    
    print(result)
    
    print(tf.argmax(result, dimension=1))
    cv2.imshow("prediction", imageOrig)
    cv2.waitKey(0)
    
    
   
   
   
