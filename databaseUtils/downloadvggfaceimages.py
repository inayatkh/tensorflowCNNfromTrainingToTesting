'''
Created on Oct 5, 2017

@author: inayat
'''

from scipy import misc
import numpy as np
from skimage import io
import time
import os
import socket
#from urllib2 import urlopen
import urllib
global f
def downloadVggfaces(vggfacesURLdir):
  socket.setdefaulttimeout(30)
  #datasetDescriptor = 'files'
  datasetDescriptor = vggfacesURLdir
  textFileNames = sorted(os.listdir(datasetDescriptor))
  person = 0
 
  _tmpdir = os.path.join(datasetDescriptor, "faces")
  if not os.path.exists(_tmpdir):
        os.makedirs(_tmpdir)
        
      
  for textFileName in textFileNames:
    if textFileName.endswith('.txt'):
      person += 1
      with open(os.path.join(datasetDescriptor, textFileName), 'rt') as f:
        lines = f.readlines()
      lastLine = int(lines[-1].split(' ')[0])
      #print("lasLine:",lastLine)
      dirName = textFileName.split('.txt')[0]
      classPath = os.path.join(datasetDescriptor,"faces", dirName)
      if not os.path.exists(classPath):
        os.makedirs(classPath)
        lastfile = 0
      else:
        files = sorted(os.listdir(classPath))
        print("files",files)
        lastfile = int(files[-1].split('.png')[0])
 
        if lastLine == lastfile:
          print(person, dirName, lastfile, "Done!")
          continue
 
      for line in lines:
        x = line.split(' ')
        fileName = x[0]
        url = x[1]
        errorLine = ''
 
        if lastfile < int(fileName):
          #print(x)
          #box = np.rint(np.array(map(float, x[2:6])))
          #box = np.array(map(float, x[2:6]))
          box = np.rint(np.array(x[2:6]).astype(float))
          #print("**********",box)
          imagePath = os.path.join(datasetDescriptor,"faces", dirName, fileName+'.png')
 
          if not os.path.exists(imagePath):
            try:
              #img = io.imread(urlopen(url,timeout = 10))
              with urllib.request.urlopen(url, timeout=10) as url_io:
                  img= io.imread(url_io)
              print("_________",img.shape)
            except Exception as e:
              errorMessage = '{}: {}'.format(url, e)
              errorLine = line
              #print("DDDDDDDDDDDDDDDDDDDDDDD")
            else:
              try:
                if img.ndim == 2:
                  img = toRgb(img)
                if img.ndim != 3:
                  raise Exception('Wrong number of image dimensions')
                hist = np.histogram(img, 255, density=True)
                if hist[0][0] > 0.9 and hist[0][254] > 0.9:
                  raise Exception('Image is mainly black or white')
                else:
                  errorMessage = 'ok!'
                  imgCropped = img[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
                  imgResized = misc.imresize(imgCropped, (256,256))
                  misc.imsave(imagePath, imgResized)
              except Exception as e:
                errorMessage = '{}: {}'.format(url, e)
                errorLine = line
            print(person,dirName,fileName,errorMessage)
        #with open("fix/"+dirName+".txt","a") as fix:
        with open(datasetDescriptor + "/"+dirName+".txt","a") as fix:
          if line != errorLine:
            fix.write(line)
      print (dirName + " Done!")
 
 
def toRgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret