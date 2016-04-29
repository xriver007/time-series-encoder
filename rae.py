import theano
from theano import tensor as T
import numpy as np
from load import *
from sklearn import datasets

trX = loadDataset1()

trX = trX.reshape( (361, 20, 1) ) # samples X time X features
trX = np.transpose(trX, (1,0,2) ) # time X samples X feats

trX = (trX - trX.min())/trX.max()

timeSteps, nSamples, nFeats = trX.shape

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def mse(y1, y2):
        return T.mean((y1 - y2) ** 2)

def encode(x_t, mem, w1, wH, w2):
    s_t = T.tanh(T.dot(x_t,w1) + T.dot(mem,wH))
    y_t = rectify(T.dot(s_t,w2))
    return [s_t, y_t]

def decode(x_t, mem, w1, wH, w2):
    s_t = T.tanh(T.dot(x_t,w1) + T.dot(mem,wH))
    y_t = T.dot(s_t,w2)
    return [s_t, y_t]
    
class rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum, rescale=5.):
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1E-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates

memSize = 10

X = T.tensor3() 
s0 = init_weights((nSamples, memSize))
s0_2 = init_weights((nSamples, memSize))

eW1 = init_weights((nFeats, memSize))
eWh = init_weights((memSize, memSize))
eW2 = init_weights((memSize,nFeats))

dW1 = init_weights((nFeats, memSize))
dWh = init_weights((memSize, memSize))
dW2 = init_weights((memSize,2))

[s, y], _ = theano.scan(encode, sequences=X, outputs_info=[s0, None], non_sequences=[eW1, eWh, eW2], strict=True)
code = y[1::2]
[s, y], _ = theano.scan(decode, sequences=code, outputs_info=[s0_2, None], non_sequences=[dW1, dWh, dW2], strict=True)

decoded = y.dimshuffle(1,0,2)
decoded = decoded.flatten(2)
decoded = decoded.dimshuffle(1,0)
decoded = decoded.reshape((timeSteps, nSamples, nFeats))

cost = mse(decoded, X)

params = [eW1, eWh, eW2, dW1, dWh, dW2, s0, s0_2]
grads = T.grad(cost, params)
opt = rmsprop(params)
updates = opt.updates(params, grads, 0.001, 0.9)

train = theano.function(inputs=[X], outputs=cost, updates=updates, allow_input_downcast=True)
encode = theano.function(inputs=[X], outputs=code, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=decoded, allow_input_downcast=True)

for i in range(5000):
    print "---------------------------"
    print i
    cost = train(trX)
    print cost

print trX[:,-1,:]
print encode(trX)[:,-1,:]
print predict(trX)[:,-1,:]