import theano
from theano import tensor as T
import numpy as np
from load import loadLinearDataset
from sklearn import datasets

trX, trY = loadLinearDataset()

trX = trX.reshape( (361, 20, 1) )
trX = np.transpose(trX, (1,0,2) )

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def foward(x_t, mem, w1, wH, w2):
    s_t = T.tanh(T.dot(x_t,w1) + T.dot(mem,wH))
    y_t = T.nnet.softmax(T.dot(s_t,w2))
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

X = T.tensor3()
s0 = theano.shared(np.zeros( (361,100), dtype=theano.config.floatX))
Y = T.imatrix()

w1 = init_weights((1, 100))
wH = init_weights((100, 100))
w2 = init_weights((100,2))

[s, y], _ = theano.scan(foward, sequences=X, outputs_info=[s0, None], non_sequences=[w1, wH, w2], strict=True)
py_x = y[-1]
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))

params = [w1, wH, w2]
grads = T.grad(cost, params)
opt = rmsprop(params)
updates = opt.updates(params, grads, 0.0001, 0.9)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(500):
    cost = train(trX, trY)
    print "----------------------------------"
    print i
    print "train acc: ", np.mean(np.argmax(trY, axis=1) == predict(trX))