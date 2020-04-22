import time
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import non_negative_factorization

import sys
sys.path.append('/home/sxchao')
from bayesmf.models.nmf import VanillaNMF, ConsensusNMF
from bayesmf.models.bmf import BayesMF, OnlineBayesMF


def workhorse(X_train, X_test, n_components, method, random_state=22690):
    if method == 'vanilla':
        W, H, err = VanillaNMF(X_train.T, n_components=n_components)
        W, H, n_iter = non_negative_factorization(X_test.T, H=H, n_components=n_components, update_H=False, 
                                                  init=None, random_state=random_state)
        
    elif method == 'consensus':
        W, H, err = ConsensusNMF(X_train.T, n_components=n_components)
        W, H, n_iter = non_negative_factorization(X_test.T, H=H, n_components=n_components, update_H=False, 
                                                  init=None, random_state=random_state)
        
    elif method == 'batch':
        factorizer = BayesMF(K = n_components)
        factorizer.fit(X_train)
        W = factorizer.transform(X_test, attr='Et').T
        H = factorizer.Eb.T
        
    elif method == 'stochastic':
        factorizer = OnlineBayesMF(K = n_components)
        factorizer.fit(X_train)
        W = factorizer.transform(X_test, attr='Et').T
        H = factorizer.Eb.T

    else:
        print('invalid method')

    return np.sqrt(np.sum((X_test.T - np.matmul(W, H)) ** 2) / X_test.size)


def run_kfold_xval(kfold=5, random_state=22690)
    idxs = np.arange(X.shape[1])
    
    if type(random_state) is int:
        np.random.seed(22690)
    np.random.shuffle(idxs)

    splits = np.split(idxs, kfold)

    methods = ['vanilla', 'consensus', 'batch', 'stochastic']
    components = [5, 10, 15, 20, 25]
    errs = {k:{v:[] for v in methods} for k in components}
    durs = {k:{v:[] for v in methods} for k in components}

    for n_components in tqdm(components):
        for k in tqdm(range(kfold)):
            idxs_train = [i for j in np.setdiff1d(np.arange(kfold), k) for i in splits[j]]
            idxs_test = splits[k]
            X_train = X[:, idxs_train]
            X_test = X[:, idxs_test]
            for method in methods:
                start = time.time()
                err = workhorse(X_train, X_test, n_components, method)
                end = time.time()
                dur = end - start
                errs[n_components][method].append(err)
                durs[n_components][method].append(dur)

    return errs, durs