import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.decomposition import non_negative_factorization


def VanillaNMF(X, n_components=15, random_state=22690):
    model = NMF(n_components=n_components, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    err = model.reconstruction_err_
    return W, H, err


def ConsensusNMF(X, n_components=15, random_state=22690, cluster_method='dbscan', interval=0, n_iters=50):
    if type(random_state) is int:
        np.random.seed(random_state)
        
    ns = np.random.randint(max(5, n_components - interval // 2), 
                           min(95, n_components + interval // 2 + 1), 
                           size=n_iters)

    cs = []
    for n in ns:
        model = NMF(n_components=n, random_state=random_state)
        W = model.fit_transform(X)
        H = model.components_
        cs.append(H)
        
    cs = np.concatenate(cs, axis=0)

    if cluster_method == 'dbscan':
        dbscan = DBSCAN(eps=3).fit(cs)
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

    W, H, n_iter = non_negative_factorization(X, H=H, n_components=n_components, update_H=False, 
                                              init=None, random_state=random_state)
    err = np.sqrt(np.sum((X - np.matmul(W, H)) ** 2) / X.shape[0])

    return W, H, err