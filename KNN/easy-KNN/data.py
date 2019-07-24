from numpy import *
import operator


def CreateDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0, 0],[0,0.1]])
    labels=["A","A","B","B"]
    return group,labels
