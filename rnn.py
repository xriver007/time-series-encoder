import theano
from theano import tensor as T
import numpy as np
from load import dataset
from sklearn import datasets

trX, teX, trY, teY = loadDataset()

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def entropy(k):
    k = clip(k)
    return T.mean(-k * T.log2(k) - (1 - k) * T.log2(1 - k))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)
 
def clip(k):
    return T.minimum(T.maximum(k, 0.00000001), 0.9999999)

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w1, wH, kH, w2):
    u = T.dot(X,w1)
    for i in range(len(wH)):
        u = clip(kH[i]) * rectify(T.dot(u, wH[i])) +  (1 - clip(kH[i]) ) * u
    return T.nnet.softmax(T.dot(u, w2))

def adam(cost, params, lr=0.0005, b1=0.1, b2=0.001, e=1e-3):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

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

X = T.fmatrix()
Y = T.fmatrix()

w1 = init_weights((784, 100))
wH = [init_weights((100, 100)) for i in range(9)] 
kH = theano.shared(floatX([0.5]*9))
w2 = init_weights((100,10))

py_x = model(X, w1, wH, kH, w2)
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y)) + entropy(kH)

preParams = [w1] + wH + [w2]
grads = T.grad(cost, preParams)
opt = rmsprop(preParams)
preUpdates = opt.updates(preParams, grads, 0.0001, 0.9)

params = [w1] + wH + [kH] + [w2]
grads = T.grad(cost, params)
opt = rmsprop(params)
updates = opt.updates(params, grads, 0.0001, 0.9)

preTrain = theano.function(inputs=[X, Y], outputs=cost, updates=preUpdates, allow_input_downcast=True)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)
	
for i in range(50):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = preTrain(trX[start:end], trY[start:end])
    print "----------------------------------"
    print i
    print "train acc: ", np.mean(np.argmax(trY, axis=1) == predict(trX))
    print "vld acc: ", np.mean(np.argmax(teY, axis=1) == predict(teX))
    print "mean weights: ", np.mean(map(lambda x: x.get_value(), wH))
    print "k's:", kH.get_value()
for i in range(50):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print "----------------------------------"
    print i
    print "train acc: ", np.mean(np.argmax(trY, axis=1) == predict(trX))
    print "vld acc: ", np.mean(np.argmax(teY, axis=1) == predict(teX))
    print "mean weights: ", np.mean(map(lambda x: x.get_value(), wH))
    print "k's:", kH.get_value()