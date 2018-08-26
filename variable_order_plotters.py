import matplotlib.pyplot as plt
from matplotlib import gridspec
from autograd import numpy as np 
import copy

def get_ylim(seq):
    smin = np.min(copy.deepcopy(seq))
    smax = np.max(copy.deepcopy(seq))
    sgap = (smax - smin)*0.1
    smin -= sgap
    smax += sgap
    return [smin, smax]

def plot_multiples(v,train=0,lab=['model','commodity','predictor'],col=['lime','m','k']):            

    fig = plt.figure(figsize = (10,5))
    gs = gridspec.GridSpec(2,1) 
    
    for i in range(2):
        ax = plt.subplot(gs[i]);
        td = int(train*i*len(v[0]))
        
        for j in range(len(v)):
            ax.plot(np.arange(td*i, np.size(v[j])),v[j][td:].flatten(),c = col[j],linewidth = 2.5,label = lab[j],zorder = i+1)
        
        title = 'model vs actual ' + '(train and test)' * (1-i) + '(test only)' * i
        ax.set_title(title)
        ax.set_xlabel('step')
        ax.set_ylim(get_ylim(v[1][td*i:]))
        ax.set_xlim(xmin = td*i)
    
        ax.axvspan(int(np.size(v[i])*train), np.size(v[i]), alpha=0.5, color='blue')
        ax.plot(np.arange(np.size(v[i])),np.ones(np.size(v[0])),c = 'b',linewidth = 2.5,label = 'test',zorder = 1)
        
        if i-1:
            ax.axvspan(0, int(np.size(v[i])*train), alpha=0.5, color='red')
            ax.plot(np.arange(np.size(v[i])),np.ones(np.size(v[0])),c = 'r',linewidth = 2.5,label = 'train',zorder = 1)
        
        ax.legend(loc = 0)
        
    plt.show()

def plot_n(v,labels):   
    fig = plt.figure(figsize = (10,2.5*len(v)))
    gs = gridspec.GridSpec(len(v),1) 
    
    for i in range(len(v)):
        ax = plt.subplot(gs[i,0]);
        ax.plot(np.arange(np.size(v[i])),v[i].flatten(),c = 'k',linewidth = 2.5)
        
        ax.set_title(labels[i])
        ax.set_xlabel('step')
        ax.set_ylim(get_ylim(v[i]))
        
    plt.show()