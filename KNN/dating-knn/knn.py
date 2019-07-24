from data_process import file_to_matrix,Nomr
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


def knn():
    """
    code for test
    :return: nums of error
    """

    rate=0.1

    data,label=file_to_matrix('./data.txt')

    norm_data=Nomr(data)

    m=norm_data.shape[0]

    num_test=int(rate*m)
    error_count=0

    for i in range(num_test):
        class_result=KNN_classify(norm_data[i:],norm_data[num_test:m,:],label[num_test:m],3)
        if (class_result!=norm_data[i]):
            error_count+=1
    return (error_count/float(num_test))
