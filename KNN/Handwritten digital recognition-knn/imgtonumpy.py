import numpy as np


def imgtonumpy(file):
    vec=np.zeros((1,1024))
    f=open(file)
    for i in range(32):
        line=f.readlines()
        for j in range(32):
            vec[0,32+i+j]=int(line[j])

    return vec


