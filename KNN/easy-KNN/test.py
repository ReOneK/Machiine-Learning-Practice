from data import CreateDataSet
from KNN import KNN_classify


data,label=CreateDataSet()
input=[0.1,0.5]
x=KNN_classify(input,data,label,2)
print(x)

