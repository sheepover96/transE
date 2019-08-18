import numpy as np

class TransE():

    def __init__(self, nentity, nrelation, margin, batch_size, lr, vector_dim):
        self.margin = margin
        self.nentity = nentity
        self.nrelation = nrelation
        self.lr = lr
        self.vector_dim = vector_dim
        self.batch_size = batch_size
    
    def _normalize(self, s):
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        return s

    def minibatch_sample(self, S, batch_size):
        for i in range(int(S.shape[0]/batch_size-1)):
            yield S[i*batch_size:(i+1)*batch_size]
        yield S[i*(batch_size-1)*batch_size:]

    def init_params(self):
        e = (2*(6/np.sqrt(self.vector_dim)))*np.random.random_sample((self.nentity, self.vector_dim)) - (6/np.sqrt(self.vector_dim))
        l = (2*(6/np.sqrt(self.vector_dim)))*np.random.random_sample((self.nentity, self.vector_dim)) - (6/np.sqrt(self.vector_dim))
        self._normalize(l)
        #l /= np.linalg.norm(l, axis=1, keepdims=True)
        params = {}
        params['e'] = e
        params['l'] = l
        return params

    def negative_sampling(self, batch):
        r = np.random.randint(2, size=batch.shape[0])
        e = np.random.randint(self.nentity, size=batch.shape[0])
        return np.vstack([batch.T,r,e]).T

    def calc_grad(self, h, l, t):
        return (self.params['e'][h] + self.params['l'][l] - self.params['e'][t])/np.linalg.norm(self.params['e'][h] + self.params['l'][l] - self.params['e'][t])

    def dist(self, h, l, t):
        return np.linalg.norm(self.params['e'][h] + self.params['l'][l] - self.params['e'][t])

    def evaluate_grad(self, batch):
        grads = {}
        grads['e'] = np.zeros((self.nentity, self.vector_dim))
        grads['l'] = np.zeros((self.nentity, self.vector_dim))
        count = {}
        count['e'] = np.zeros(self.nentity)
        count['l'] = np.zeros(self.nentity)
        T = self.negative_sampling(batch)
        for h,l,t,rnd,e in T:
            if rnd == 1: h2 = h; t2 = e
            else: h2 = e; t2 = t
            if self.dist(h,l,t) + self.margin - self.dist(h2,l,t2):
                grad1 = self.calc_grad(h,l,t)
                grad2 = self.calc_grad(h2,l,t2)
                grads['e'][h] += grad1
                grads['e'][t] += -grad1
                grads['e'][h2] += -grad2
                grads['e'][t2] += grad2
                grads['l'][l] += grad1 - grad2
                count['e'][h] += 1
                count['e'][t] += 1
                count['e'][h2] += 1
                count['e'][t2] += 1
                count['l'][l] += 1
        count['e'][count['e']==0] = 1
        count['l'][count['l']==0] = 1
        grads['e'] = (grads['e'].T/count['e']).T # 1/n
        grads['l'] = (grads['l'].T/count['l']).T # 1/n
        return grads

    def fit(self, S, nepochs=1000, validationset=None):
        self.params = self.init_params()

        for i in range(nepochs):
            print(i)
            for batch in self.minibatch_sample(S, self.batch_size):
                self.params['e'] = self._normalize(self.params['e'])
                grads = self.evaluate_grad(batch)
                self.params['e'] -= self.lr*grads['e']
                self.params['l'] -= self.lr*grads['l']

