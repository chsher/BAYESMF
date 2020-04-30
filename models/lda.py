# adapted from @blei-lab onlineldavb

import numpy as np
from scipy import special
from sklearn.base import BaseEstimator, TransformerMixin


def _compute_expectations(a, return_exp=True):
    '''
    Computes the expectation of the log of x_n ~ Dir(a_n) \forall n \in [N].
    E[log(x_n)|a_n] = digamma(a_n) - digamma(sum_m a_{nm}) \forall m \in [M]. 

    Parameters
    ----------
    a : array-like, shape (N x M)
    return_exp : bool, whether to return the exponential of Elogx

    Returns
    -------
    Elogx : array-like, shape (N x M)
    exp^{Elogx} : if return_exp is True, array-like, shape (N x M)
    '''

    if len(a.shape) == 1:
        Elogx = special.psi(a) - special.psi(np.sum(a))
    else:    
        Elogx = special.psi(a) - special.psi(np.sum(a, axis=1)[:, np.newaxis])

    if return_exp:
        return Elogx, np.exp(Elogx)
    else:
        return Elogx

 
 ITER_STMT = 'Iter: {0:d}, Bound: {1:.2f}, Change: {2:.5f}'

class LDA(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 K=15, 
                 max_iters=100, 
                 tolerance=0.001, 
                 smoothness=100, 
                 random_state=22690, 
                 verbose=False,
                 **kwargs):
        
        self.K = K
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        self.alpha = float(kwargs.get('alpha', 0.1))
        self.eta = float(kwargs.get('eta', 0.1))

    # local
    def _init_qtheta(self, D):
        self.gamma = np.random.gamma(self.smoothness, 
                                     scale = 1.0 / self.smoothness, 
                                     size=(D, self.K))
        self.Elogt, self.eElogt = _compute_expectations(self.gamma)
        
    # global
    def _init_qbeta(self, V):
        self.lambd = np.random.gamma(self.smoothness, 
                                     scale = 1.0 / self.smoothness, 
                                     size=(self.K, V))
        self.Elogb, self.eElogb = _compute_expectations(self.lambd)

    def fit(self, X):
        D, V = X.shape
        self._init_qtheta(D)
        self._init_qbeta(V)
        self._update(X)
        return self

    def transform(self, X, attr='Et'):
        if not hasattr(self, 'Eb'):
            raise ValueError('No beta initialized.')

        D, V = X.shape
        if not self.Eb.shape[1] == V:
            raise ValueError('Feature dim mismatch.')

        self._init_qtheta(D)
        self._update(X, update_beta=False)
        return getattr(self, attr)

    def _update(self, X, update_beta=True):
        elbo_old = -np.inf
        for i in range(self.max_iters):
            self._update_theta(X)

            if update_beta:
                self._update_beta(X)

            elbo_new = self._bound(X)
            chg = (elbo_new - elbo_old) / abs(elbo_old)
            
            if self.verbose and i % 10 == 0:
                print(ITER_STMT.format(i, elbo_new, chg))

            if chg < self.tolerance:
                break

            elbo_old = elbo_new

    def _update_theta(self, X, reinit_theta=True):
        D = X.shape[0]
            
        if reinit_theta:
            self._init_qtheta(D)

        for d in range(D):
            counts_d = X[d, :]

            gamma_d = self.gamma[d, :]
            eElogt_d = self.eElogt[d, :]

            for i in range(self.max_iters):
                gamma_old = gamma_d

                phi_d = eElogt_d * np.dot(counts_d / self._phisum(d, eElogt_d=eElogt_d), self.eElogb.T) 
                gamma_d = self.alpha + phi_d

                chg = np.mean(abs(gamma_d - gamma_old))
                if chg < self.tolerance:
                    break

                eElogt_d, _ = _compute_expectations(gamma_d)

            self.gamma[d, :] = gamma_d

        self.Elogt, self.eElogt = _compute_expectations(self.gamma)
        
    def _update_beta(self, X): 
        D, V = X.shape
        total = np.zeros((self.K, V))

        for d in range(D):
            counts_d = X[d, :]
            total += np.outer(self.eElogt[d, :], counts_d / self._phisum(d))

        self.lambd = self.eta + total * self.eElogb
        self.Elogb, self.eElogb = _compute_expectations(self.lambd)

    def _phisum(self, d, eElogt_d=None):
        ''' 
        Returns sum_k exp{Elogt_dk} + exp{Elogb_k}
        '''
        if eElogt_d is not None:
            return np.dot(eElogt_d, self.eElogb) + 1e-100
        else:
            return np.dot(self.eElogt[d, :], self.eElogb) + 1e-100

    def _bound(self, X):
        D = X.shape[0]
        bound = 0

        # E[E[log p(docs | theta, z, beta)] + E[log p(z | theta)] - log q(z)]
        for d in range(D):
            counts_d = X[d, :]
            Eloglik_d = self.Elogb + self.Elogt[d, :][:, np.newaxis]
            # np.outer(a,b):
            #  [ [a0*b0  a0*b1 ... a0*bV ]
            #    [a1*b0    .
            #    [ ...          .
            #    [aK*b0            aK*bV ] ]
            phi_d = np.outer(self.eElogt[d, :], 1.0 / self._phisum(d)) * self.eElogb 
            bound += special.logsumexp(counts_d[np.newaxis, :] * phi_d * (Eloglik_d + np.log(phi_d)))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        bound += np.sum((self.alpha - self.gamma) * self.Elogt)
        bound += np.sum(special.gammaln(self.gamma))
        bound -= np.sum(special.gammaln(np.sum(self.gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        bound += np.sum((self.eta - self.lambd) * self.Elogb)
        bound += np.sum(special.gammaln(self.lambd))
        bound -= np.sum(special.gammaln(np.sum(self.lambd, 1)))

        return bound
