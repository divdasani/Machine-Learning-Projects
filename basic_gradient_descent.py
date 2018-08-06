import autograd.numpy as np   
from autograd import value_and_grad 

class Setup:
    
    def __init__(self,x,y,**kwargs):
        self.x = x
        self.y = y

        self.weight_histories = []
        self.cost_histories = []
        self.count_histories = []
    
    def model(self,x,w):    
        # stack a 1 onto the top of each input point
        o = np.ones((1,np.shape(x)[1]))
        x = np.vstack((o,x))

        # compute linear combination and return
        a = np.dot(x.T,w)
        return a

    def least_squares(self,w):    
        cost = np.sum((self.model(self.x,w) - self.y)**2)
        return cost/float(len(self.y))
    
    def softmax(self,w):
        cost = np.sum(np.log(1 + np.exp(-self.y*self.model(self.x,w))))
        return cost/float(len(self.y))
    
    def multiclass_softmax(self, w):
        all_evals = self.model(self.x,w)

        a = np.log(np.sum(np.exp(all_evals),axis = 1)) 
        b = all_evals[np.arange(len(self.y)),self.y.astype(int).flatten()]
        cost = np.sum(a - b)

        # add optional regularizer
        lam = 0
        cost = cost + lam*np.linalg.norm(w[1:,:],'fro')**2
        
        return cost/float(len(self.y))
    
    def counting_cost(self,w):
        cost = np.sum((np.sign(self.model(self.x,w)) - self.y)**2)
        return 0.25*cost 
    
    def multiclass_counting_cost(self,w):  
        all_evals = self.model(self.x,w)

        y_predict = (np.argmax(all_evals,axis = 1))[:,np.newaxis]
        count = np.sum(np.abs(np.sign(self.y - y_predict)))

        return count
    
    def gradient_descent(self,g,alpha_choice,max_its,w):
        gradient = value_and_grad(g)

        weight_history = []      
        cost_history = []        
        alpha = 0

        for k in range(1,max_its+1):
            # check if diminishing steplength rule used
            if alpha_choice == 'diminishing':
                alpha = 1/float(k)
            else:
                alpha = alpha_choice

            cost_eval,grad_eval = gradient(w)
            weight_history.append(w)
            cost_history.append(cost_eval)

            # take gradient descent step
            w = w - alpha*grad_eval

        # collect final weights
        weight_history.append(w)

        cost_history.append(g(w))  
        return weight_history,cost_history
    
    def standard_normalizer(self,x):
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   

        normalizer = lambda data: (data - x_means)/x_stds
        inv_normalizer = lambda data: (data * x_stds) + x_means

        return normalizer, inv_normalizer
    
    def normalized_gradient_descent(self,g,alpha_choice,max_its,w):
        normalizer, inv_normalizer = self.standard_normalizer(self.x)
        x_normalized = normalizer(self.x)
        self.x = x_normalized
        weight_history, cost_history = self.gradient_descent(g,alpha_choice,max_its,w)
        return weight_history, cost_history, x_normalized
        

    