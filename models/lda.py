# adapted from @blei-lab onlineldavb

import numpy as np
from scipy import special
from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, TransformerMixin


ITER_STMT = 'Iter: {0:d}, Bound: {1:.2f}, Change: {2:.5f}'
EPOCH_STMT = 'Epoch: {0:d}'
MINIBATCH_STMT = 'Minibatch: {0:d}, Bound: {1:.2f}'
EPOCH_SUMMARY_STMT = 'Epoch: {0:d}, Avg Bound: {1:.2f}, Change: {2:.5f}'


def _compute_expectations(a, return_exp=True):
    '''
    Computes the expectation of [the log of] x_n ~ Dir(a_n).
    E[x_n] = \frac{a_n}{sum_m a_{nm}}.
    E[log(x_n)|a_n] = digamma(a_n) - digamma(sum_m a_{nm}). 

    Parameters
    ----------
    a : array-like, shape (N x M)
    return_exp : bool, whether to return the exponential of Elogx

    Returns
    -------
    Ex : array-like, shape (N x M)
    Elogx : array-like, shape (N x M)
    exp^{Elogx} : if return_exp is True, array-like, shape (N x M)
    '''
    if len(a.shape) == 1:
        Ex = a / np.sum(a)
        Elogx = special.psi(a) - special.psi(np.sum(a))
    else:    
        Ex = a / np.sum(a, axis=1)[:, np.newaxis]
        Elogx = special.psi(a) - special.psi(np.sum(a, axis=1)[:, np.newaxis])

    if return_exp:
        return Ex, Elogx, np.exp(Elogx)
    else:
        return Ex, Elogx
    
    
class LDA(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 K=15, 
                 max_iters=100, 
                 tolerance=0.0005, 
                 smoothness=100, 
                 random_state=22690, 
                 verbose=False,
                 init=None,
                 **kwargs):
        self.K = K
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose
        self.init = init

        if type(self.random_state) is int:
            np.random.seed(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        self.alpha = float(kwargs.get('alpha', 0.1))
        self.eta = float(kwargs.get('eta', 0.1))

    # local
    def _init_qtheta(self, D):
        if self.init is None:
            self.gamma = np.random.gamma(self.smoothness, 
                                         scale = 1.0 / self.smoothness, 
                                         size=(D, self.K))
        elif self.init == 'nmf':
            self.gamma = self.W * np.sum(self.W, axis=1)
            
        self.Et, self.Elogt, self.eElogt = _compute_expectations(self.gamma)
        
    # global
    def _init_qbeta(self, V):
        if self.init is None:
            self.lambd = np.random.gamma(self.smoothness, 
                                         scale = 1.0 / self.smoothness, 
                                         size=(self.K, V))
        elif self.init == 'nmf':
            self.lambd = self.H * np.sum(self.H, axis=1)
            
        self.Eb, self.Elogb, self.eElogb = _compute_expectations(self.lambd)

    def fit(self, X):
        D, V = X.shape
        
        if self.init == 'nmf':
            model = NMF(n_components=self.K, random_state=self.random_state)
            self.W = model.fit_transform(X)
            self.H = model.components_

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

        if self.init == 'nmf':
            model = NMF(n_components=self.K, random_state=self.random_state)
            self.W = model.fit_transform(X)
            
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
            
            if self.verbose and update_beta:
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

                _, _, eElogt_d = _compute_expectations(gamma_d)

            self.gamma[d, :] = gamma_d

        self.Et, self.Elogt, self.eElogt = _compute_expectations(self.gamma)
        
    def _update_beta(self, X): 
        D, V = X.shape
        total = np.zeros((self.K, V))

        for d in range(D):
            counts_d = X[d, :]
            total += np.outer(self.eElogt[d, :], counts_d / self._phisum(d))

        self.lambd = self.eta + total * self.eElogb
        self.Eb, self.Elogb, self.eElogb = _compute_expectations(self.lambd)

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

        # E[ E[ log p(docs | theta, z, beta)] + E[log p(z | theta) ] - log q(z) ]
        for d in range(D):
            counts_d = X[d, :]
            Eloglik_d = self.Elogb
            phi_d = np.outer(self.eElogt[d, :], 1.0 / self._phisum(d)) * self.eElogb 
            zterms_d = self.Elogt[d, :][:, np.newaxis] - np.log(phi_d)
            bound += special.logsumexp(counts_d[np.newaxis, :] * phi_d * (Eloglik_d + zterms_d))

        # E[ log p(theta | alpha) - log q(theta | gamma) ]
        bound += np.sum((self.alpha - self.gamma) * self.Elogt)
        bound += np.sum(special.gammaln(self.gamma))
        bound -= np.sum(special.gammaln(np.sum(self.gamma, 1)))

        # E[ log p(beta | eta) - log q(beta | lambda) ]
        bound += np.sum((self.eta - self.lambd) * self.Elogb)
        bound += np.sum(special.gammaln(self.lambd))
        bound -= np.sum(special.gammaln(np.sum(self.lambd, 1)))

        return bound
    
    
class StochasticLDA(LDA):
    def __init__(self, 
                 K=15, 
                 n_epochs=5,
                 minibatch_size=100,
                 shuffle=True,
                 max_iters=100, 
                 tolerance=0.0005, 
                 smoothness=100, 
                 random_state=22690, 
                 verbose=False,
                 init=None,
                 **kwargs):
        self.K = K
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.shuffle = shuffle
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose
        self.init = init

        if type(self.random_state) is int:
            np.random.seed(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        self.alpha = float(kwargs.get('alpha', 0.1))
        self.eta = float(kwargs.get('eta', 0.1))
        self.tau = float(kwargs.get('tau', 1.))
        self.kappa = float(kwargs.get('kappa', 0.6))

    def fit(self, X):
        D, V = X.shape

        self._scale = float(D) / self.minibatch_size
        
        if self.init == 'nmf':
            model = NMF(n_components=self.K, random_state=self.random_state)
            self.W = model.fit_transform(X)
            self.H = model.components_
            
        self._init_qbeta(V)
        
        self.bound = []

        elbo_old = -np.inf
        for e in range(self.n_epochs):
            if self.verbose:
                print(EPOCH_STMT.format(e + 1))

            idxs = np.arange(D)
            if self.shuffle:
                np.random.shuffle(idxs)

            elbo_new = 0
            X_shuffled = X[idxs, :]
            for (t, start) in enumerate(range(0, D, self.minibatch_size), 1):
                self.set_step_size(t=t)

                end = min(start + self.minibatch_size, D)
                minibatch = X_shuffled[start:end, :]
                
                self.partial_fit(minibatch)

                elbo = self._stochastic_bound(minibatch)
                elbo_new += elbo

                if self.verbose:
                    print(MINIBATCH_STMT.format(t, elbo))

                self.bound.append(elbo)

            elbo_new /= t
            chg = (elbo_new - elbo_old) / abs(elbo_old)

            if self.verbose:
                print(EPOCH_SUMMARY_STMT.format(e + 1, elbo_new, chg))

            if chg < self.tolerance:
                break

            elbo_old = elbo_new

        return self

    def partial_fit(self, X):
        self.transform(X)
        
        D, V = X.shape
        total = np.zeros((self.K, V))

        for d in range(D):
            counts_d = X[d, :]
            total += np.outer(self.eElogt[d, :], counts_d / self._phisum(d))

        lambd_new = self.eta + self._scale * total * self.eElogb
        self.lambd = (1 - self.rho) * self.lambd + self.rho * lambd_new
        self.Eb, self.Elogb, self.eElogb = _compute_expectations(self.lambd)

        return self
        
    def set_step_size(self, t=None):
        if t is not None:
            self.rho = (t + self.tau)**(-self.kappa)
        else:
            raise ValueError('Cannot set step size.')

        return self

    def _stochastic_bound(self, X):
        D = X.shape[0]
        bound = 0

        for d in range(D):
            counts_d = X[d, :]
            Eloglik_d = self.Elogb
            phi_d = np.outer(self.eElogt[d, :], 1.0 / self._phisum(d)) * self.eElogb 
            zterms_d = self.Elogt[d, :][:, np.newaxis] - np.log(phi_d)
            bound += special.logsumexp(counts_d[np.newaxis, :] * phi_d * (Eloglik_d + zterms_d))

        bound += np.sum((self.alpha - self.gamma) * self.Elogt)
        bound += np.sum(special.gammaln(self.gamma))
        bound -= np.sum(special.gammaln(np.sum(self.gamma, 1)))

        bound *= self._scale

        bound += np.sum((self.eta - self.lambd) * self.Elogb)
        bound += np.sum(special.gammaln(self.lambd))
        bound -= np.sum(special.gammaln(np.sum(self.lambd, 1)))
        
        return bound
