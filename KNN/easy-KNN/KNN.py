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
