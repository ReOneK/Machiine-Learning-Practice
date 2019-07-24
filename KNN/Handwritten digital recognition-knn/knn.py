from os import listdir
from imgtonumpy import imgtonumpy
import numpy as np
import operator


def KNN_classify(input,dataset,labels,k):
    dataset_size=dataset.shape[0]

    #compute distance
    distance1=np.tile(input,(dataset_size,1))-dataset
    distance2=distance1**2
    distance3=distance2.sum(axis=1)
    distance=distance3**0.5

    sorted_dis=distance.argsort()
    classCount={}

    for i in range(k):
        label=labels[sorted_dis[i]]
        classCount[label]=classCount.get(label,0)+1
    sort_class_count=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sort_class_count[0][0]


def handwriteclass():
    labels=[]

    train_data=listdir('./data')
    m=len(train_data)
    train_mat=np.zeros((1,1024))
    for i in range(m):
        filename1=train_data[i]
        filename=filename1.split('.')[0]
        label=int(filename.split('_')[0])
        labels.append(label)
        train_mat[i,:]=imgtonumpy('./data/%s' %filename1)

    testFile = 'data/0_0.txt'  # iterate through the test set
    errorCount = 0.0
    fileNameStr = testFile
    fileStr = fileNameStr.split('.')[0]  # take off .txt
    classNumStr = int(fileStr.split('_')[0])
    vectorUnderTest = imgtonumpy('input/2.KNN/testDigits/%s' % fileNameStr)
    classifierResult = KNN_classify(vectorUnderTest, train_mat, labels, 3)
    print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))

    if classNumStr != classifierResult:
        errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
