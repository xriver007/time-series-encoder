import numpy as np

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def linearClassifier(x):
    y = map(lambda i: int(i[0] > 4) * (i[-1] < 21), x)
    y = one_hot(y,2)
    return y

def loadDataset1():
    #plain linear dataset, mainly for toy examples
    x = np.empty( (361,20) )
    index = 0
    for i in xrange(1,20):
        for j in xrange(1, 20):
            dataPoint = np.linspace(i, i + j, num=j+1)
            dataPoint = np.append(dataPoint,np.zeros(20 - len(dataPoint)))
            x[index] = dataPoint
            index += 1
    return x

def loadDataset2():
    x = np.empty( (361,20) )
    index = 0
    for i in xrange(1,20):
        for j in xrange(1, 20):
            dataPoint = np.linspace(i, i + j, num=j+1) ** 2
            dataPoint = np.append(dataPoint,np.zeros(20 - len(dataPoint)))
            x[index] = dataPoint
            index += 1
    return x

def loadLabeledDataset1():
    x = loadDataset1()
    y = linearClassifier(x)
    return [x,y]
