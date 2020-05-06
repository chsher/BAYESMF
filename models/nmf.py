import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import non_negative_factorization


def VanillaNMF(X, n_components=15, random_state=22690):
    '''
    Computes non-negative matrices W, H whose product approximates X
    by minimizing the Forbenius norm b/t X, WH via coordinate descent.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int, number of latent components
    random_state : int or None
    
    Returns
    -------
    W: array-like, shape (n_samples, n_components)
    H: array-like, shape (n_components, n_features)
    rmse: float
    '''
    model = NMF(n_components=n_components, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    rmse = mean_squared_error(X, np.matmul(W,H), squared=False)
    return W, H, rmse


def ConsensusNMF(X, n_components=15, random_state=22690, cluster_method='dbscan', eps=3, interval=0, n_iters=50):
    '''
    Computes non-negative matrices W, H whose product approximates X
    (where H is defined as the cluster centroids of multiple NMF runs)
    by minimizing the Forbenius norm b/t X, WH via coordinate descent.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int, number of latent components
    random_state : int or None
    cluster_method : string, ['dbscan', 'kmeans']
    eps : float, distance between neighbors, used in dbscan
    interval : int, range of n_components on which to run nmf
    n_iters : int, number of nmf runs
    
    Returns
    -------
    W: array-like, shape (n_samples, n_components)
    H: array-like, shape (n_components, n_features)
    rmse: float
    '''
    if type(random_state) is int:
        np.random.seed(random_state)
        
    # n_components, random_state for every iter    
    ns = np.random.randint(max(5, n_components - interval // 2), 
                           min(95, n_components + interval // 2 + 1), 
                           size=n_iters)
    rs = np.random.choice(np.arange(99999), size=n_iters, replace=False)
    
    cs = []
    for n,r in ns,rs:
        model = NMF(n_components=n, random_state=r)
        W = model.fit_transform(X)
        H = model.components_
        cs.append(H)
        
    cs = np.concatenate(cs, axis=0)

    if cluster_method == 'dbscan':
        dbscan = DBSCAN(eps=eps).fit(cs)
        counts = Counter(dbscan.labels_)

        idxs = sorted(counts, key=counts.get, reverse=True)
        if -1 in idxs:
            idxs.remove(-1)

        assert len(idxs) >= n_components, 'insufficient clusters: %d' % len(idxs)

        H = np.ndarray((n_components, X.shape[1]))
        for i,idx in enumerate(idxs[:n_components]):
            H[i, :] = np.mean(cs[dbscan.labels_ == idx, :], axis=0)

    elif cluster_method == 'kmeans':
        kmeans = KMeans(n_clusters=n_components).fit(cs)
        H = kmeans.cluster_centers_
        H[H < 0] = 0.0
        
    else:
        print('invalid clustering method')

    W, H, _ = non_negative_factorization(X, H=H, n_components=n_components, update_H=False, 
                                              init=None, random_state=random_state)
    rmse = mean_squared_error(X, np.matmul(W,H), squared=False)

    return W, H, rmse
