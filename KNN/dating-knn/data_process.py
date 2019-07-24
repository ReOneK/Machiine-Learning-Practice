import numpy as np


def file_to_matrix(file):
    """
    将文本数据转化为numpy数据
    :param file: 输入文本
    :return: data,label
    """
    f=open(file)
    numslines=len(f.readlines())

    data_mat=np.zeros(numslines,3)

    label_mat=[]
    index=0
    for line in f.readlines():
        line=line.strip()
        line=line.split('\t')

        data_mat[index,:]=line[0:3]

        label_mat=label_mat.append(line[-1])

        index+=1
    return data_mat,label_mat

def Nomr(dataset):
    """
    归一化数据
    :param dataset: 数据集
    :return: 归一化后的数据
    """
    #min_val,max_val,diff的维数都是（1，3），其中的参数0表示的是对列求值
    min_val=dataset.min(0)
    max_val=dataset.max(0)

    diff=max_val-min_val
    m=dataset.shape[0]
    normdataset=dataset-np.tile(min_val,(m,1))

    normdataset=normdataset/np.tile(diff,(m,1))

    return normdataset


