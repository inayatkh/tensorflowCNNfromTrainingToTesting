'''
Created on Sep 30, 2017

@author: inayat
'''

import numpy as np

def getIntLabels(clsNamesDict, strLabels):
    
    intLabels= None
    for lbl in strLabels :
        if intLabels is None:
            intLabels = clsNamesDict[lbl]
        else:
            intLabels=np.vstack([intLabels, clsNamesDict[lbl]])
    
    return intLabels  