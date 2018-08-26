import autograd.numpy as np   
from autograd import value_and_grad 
from autograd.misc.flatten import flatten_func

class Setup:
    
    def __init__(self,v,ph,ma,train_ratio,**kwargs):
        self.v = v
        self.ph = ph
        self.ma = ma
        train_ratio = min(train_ratio,1)
        self.train_limit = int(round(train_ratio*len(v[0]),0))

        self.weight_histories = []
        self.cost_histories = []
        self.count_histories = []
        
    def ma_builder(self, s):
        add = 0.0
        cur = 0
        a = []

        for i in range(len(s)):
            cur = s[i]
            add += cur
            if i < self.ma-1:
                a.append(cur)
            else:
                a.append(add/self.ma)
                add -= s[i-self.ma+1]

        return np.array(a).T

    def standard_normalizer(self,v):

        x_means = np.mean(v,axis = 1)[:,np.newaxis]
        x_stds = np.std(v,axis = 1)[:,np.newaxis]   
        x_stds = np.maximum(x_stds, 10**(-7))
        norm = lambda data: (data - x_means)/x_stds
        inv_norm = lambda data: (data * x_stds) + x_means

        return norm(v), [norm, inv_norm]

    def system_model(self,w,v_t):
        add = 0
        for i in range(len(w)):
            add+= w[i] * v_t[i]
        return add

    def loop(self,w,v,train=0):
        if train == 0:
            train = self.train_limit
        s_predict = []
        s_hat = 0
        excl = max(self.ph,self.ma)
        for t in range(len(v[0])):
            if t < excl or t > train:
                s_hat = v[0][t]
            else:
                v_t = [1]
                for i in range(1,len(w)):
                    v_t.append(v[i][t-self.ph])
                s_hat = self.system_model(w,v_t)
            s_predict.append(s_hat)
        return np.array(s_predict).T

    def least_squares(self,w,v):
        s_predict = self.loop(w,v)
        cost = np.sum(([s_predict] - v[0])**2)
        return cost/float(len(v[0])-self.ph)

    def gradient_descent(self,g,alpha_choice,max_its,w,v): 

        g_flat, unflatten, w = flatten_func(g, w)
        grad = value_and_grad(g_flat)

        w_hist = [unflatten(w)]
        train_hist = [g_flat(w,v)]

        alpha = 0
        for k in range(1,max_its+1):
            print('iteration: ', k, end = "\r")
            alpha = 0
            if alpha_choice == 'diminishing':
                alpha = 1/float(k)
            else:
                alpha = alpha_choice

            cost_eval,grad_eval = grad(w,v)
            grad_eval.shape = np.shape(w)

            w = w - alpha*grad_eval

            train_cost = g_flat(w,v)

            w_hist.append(unflatten(w))
            train_hist.append(train_cost)

        return w_hist,train_hist