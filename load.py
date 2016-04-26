import numpy as np


def linearClassifier(x):
    y = map(lambda i: int(i[0] > 4) * (i[-1] < 21), x)
    return y

def loadLinearDataset():
    x = []
    for i in xrange(1,20):
        for j in xrange(1, 20):
            dataPoint = np.linspace(i, i + j, num=j+1)
            x.append(dataPoint)
            
    y = linearClassifier(x)
    return [x,y]