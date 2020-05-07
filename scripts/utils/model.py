import os
import time
from tqdm import tqdm
from tqdm.auto import trange

import numpy as np
from numpy import inf
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import non_negative_factorization

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from bayesmf.models.nmf import VanillaNMF, ConsensusNMF
from bayesmf.models.lda import LDA, StochasticLDA
from bayesmf.models.bmf import BMF, StochasticBMF
from bayesmf.models.cmf import CMF, StochasticCMF
    
    
def workhorse(X_train, X_test, n_components, model, algorithm, random_state=22690, init=None):
    '''
    model : ['nmf', 'lda', 'bmf', 'cmf']
    algorithm : ['vanilla', 'consensus', 'batch', stochastic']
    '''
    if model == 'nmf':
        if algorithm == 'vanilla':
            W, H, err = VanillaNMF(X_train.T, n_components=n_components, random_state=random_state)
        elif algorithm == 'consensus':
            W, H, err = ConsensusNMF(X_train.T, n_components=n_components, random_state=random_state)
        try:
            W, H, n_iter = non_negative_factorization(X_test.T, H=H, n_components=n_components, 
                                                      update_H=False, init=None, 
                                                      random_state=random_state)
        except:
            print('error: nmf')
        
    elif model == 'lda':
        if algorithm == 'batch':
            factorizer = LDA(K=n_components, random_state=random_state, init=init)
        elif algorithm == 'stochastic':
            factorizer = StochasticLDA(K=n_components, random_state=random_state, init=init)
        try:
            factorizer.fit(X_train.T) # V x D -> D x V
            W = factorizer.transform(X_test.T, attr='Et') * np.sum(X_test, axis=0)[:,np.newaxis] # D x K
            H = factorizer.Eb # K x V
        except:
            print('error: lda')
        
    elif model == 'bmf':
        if algorithm == 'batch':
            factorizer = BMF(K=n_components, random_state=random_state, init=init)
        elif algorithm == 'stochastic':
            factorizer = StochasticBMF(K=n_components, random_state=random_state, init=init)
        try:
            factorizer.fit(X_train) # V x D
            W = factorizer.transform(X_test, attr='Et').T # K x D -> D x K
            H = factorizer.Eb.T # V x K -> K x V
        except:
            print('error: bmf')
        
    elif model == 'cmf':
        if algorithm == 'batch':
            factorizer = CMF(K=n_components, m=n_components-5, random_state=random_state, init=init, 
                             kwargs={'c0':0.05 * X_train.shape[0]})
        elif algorithm == 'stochastic':
            factorizer = StochasticCMF(K=n_components, m=n_components-5, random_state=random_state, init=init, 
                                       kwargs={'c0':0.05 * X_train.shape[0]})
        try:
            factorizer.fit(X_train) # V x D
            l = factorizer.transform(X_test, attr='l') # K x m
            W = np.exp(factorizer.alpha[np.newaxis, :] + np.dot(l, factorizer.u.T)).T # K x D -> D x K
            H = factorizer.Eb.T # V x K -> K x D
        except:
            print('error: cmf')

    else:
        print('invalid model')

    try:
        X_pred = np.matmul(W, H)
        X_pred[X_pred == inf] = 0.0
        X_pred[X_pred == -inf] = 0.0
        X_pred[np.isnan(X_pred)] = 0.0
        err = mean_squared_error(X_test.T, X_pred, squared=False)
    except:
        err = 0.0
        
    return err


def run_kfold_xval(X, kfold=5, random_state=22690, init=None,
                   components = [5, 10, 15, 20, 25], 
                   methods = ['nmf-vanilla', 'nmf-consensus', 
                              'lda-batch', 'lda-stochastic',
                              'bmf-batch', 'bmf-stochastic',
                              'cmf-batch', 'cmf-stochastic']):
    idxs = np.arange(X.shape[1])
    
    if type(random_state) is int:
        np.random.seed(22690)
    np.random.shuffle(idxs)

    splits = np.split(idxs, kfold)

    errs = {k:{v:[] for v in methods} for k in components}
    durs = {k:{v:[] for v in methods} for k in components}

    for nc in trange(len(components), desc='k-soln'):
        n_components = components[nc]
        
        for k in trange(kfold, desc='cv'):
            idxs_train = [i for j in np.setdiff1d(np.arange(kfold), k) for i in splits[j]]
            idxs_test = splits[k]
            X_train = X[:, idxs_train]
            X_test = X[:, idxs_test]
            
            for m in trange(len(methods), desc='method'):
                method = methods[m]
                model = method.split('-')[0]
                algorithm = method.split('-')[1]
                
                start = time.time()
                err = workhorse(X_train, X_test, n_components, model, algorithm, 
                                random_state=random_state, init=init)
                end = time.time()
                dur = end - start
                print('Method: {0}, RMSE: {1:.3f}, Dur: {2:.3f}'.format(method, err, dur))
                
                errs[n_components][method].append(err)
                durs[n_components][method].append(dur)

    return errs, durs
