import numpy as np
from numpy import inf
from scipy import special
from scipy.stats import norm
from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, TransformerMixin

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from bayesmf.models.bmf import _compute_expectations, _gamma_term


ITER_STMT = 'Iter: {0:d}, Bound: {1:.2f}, Change: {2:.5f}'
EPOCH_STMT = 'Epoch: {0:d}'
MINIBATCH_STMT = 'Minibatch: {0:d}, Bound: {1:.2f}'
EPOCH_SUMMARY_STMT = 'Epoch: {0:d}, Avg Bound: {1:.2f}, Change: {2:.5f}'
                        
        
class CMF(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 K=15, 
                 m=10,
                 num_steps=1,
                 step_size=1e-05,
                 max_iters=100, 
                 tolerance=0.0005, 
                 smoothness=100, 
                 random_state=22690, 
                 verbose=False,
                 init=None,
                 **kwargs):
        self.K = K
        self.m = m
        self.num_steps = num_steps
        self.step_size = step_size
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
        self.c0 = float(kwargs.get('c0', 0.1))
        self.sigma = float(kwargs.get('sigma', 1.0 / self.m))
        
    def _init_qbeta(self, V):
        if self.init is None:
            self.g = np.random.gamma(self.smoothness, 
                                     scale = 1.0 / self.smoothness, 
                                     size=(V, self.K))
        elif self.init == 'nmf':
            self.g = np.random.gamma((self.H.T + 1.0) * self.smoothness, 
                                     scale = np.ones((V, self.K)) / self.smoothness)
            
        self.h = np.random.gamma(self.smoothness, 
                                 scale = 1.0 / self.smoothness, 
                                 size=(V, self.K))
        
        self.Eb, self.Elogb = _compute_expectations(self.g, self.h)

    def _init_ql(self, D):
        if self.init is None:
            mean = np.zeros(self.m)
            cov = self.sigma * np.eye(self.m)
            self.l = np.random.multivariate_normal(mean, cov, self.K)# + np.random.random() * D
        elif self.init == 'nmf':
            self.l = self.H_m.T
            
    def _init_qu(self, D):
        if self.init is None:
            mean = np.zeros(self.m)
            cov = np.eye(self.m)
            self.u = np.random.multivariate_normal(mean, cov, D)
        elif self.init == 'nmf':
            self.u = self.W_m

    def fit(self, X):
        V, D = X.shape
        
        if self.init == 'nmf':
            model = NMF(n_components=self.K, random_state=self.random_state)
            self.W = model.fit_transform(X.T)        # n_samples, n_components
            self.H = model.components_               # n_components, n_features
            model_m = NMF(n_components=self.m, random_state=self.random_state)
            self.W_m = model_m.fit_transform(self.W) # n_samples, m
            self.H_m = model_m.components_           # m, n_components
            
        self._init_qbeta(V)
        self._init_ql(D)
        self._init_qu(D)

        self._update(X)

        return self

    def transform(self, X, attr='l'):
        if not hasattr(self, 'Eb'):
            raise ValueError('No beta initialized.')

        V, D = X.shape
        if not self.Eb.shape[0] == V:
            raise ValueError('Feature dim mismatch.')

        if self.init == 'nmf':
            model = NMF(n_components=self.K, random_state=self.random_state)
            self.W = model.fit_transform(X.T)
            model_m = NMF(n_components=self.m, random_state=self.random_state)
            self.W_m = model_m.fit_transform(self.W)
            
        self._init_qu(D)

        self._update(X, update_globals=False)

        return getattr(self, attr)

    def _update(self, X, update_globals=True):
        elbo_old = -np.inf
        
        for i in range(self.max_iters):
            self._update_alpha(X)
            self._update_qu(X)

            if update_globals:
                self._update_ql(X)
                self._update_beta(X)

            elbo_new = self._bound(X)
            chg = (elbo_new - elbo_old) / abs(elbo_old)
            
            if self.verbose:
                print(ITER_STMT.format(i, elbo_new, chg))

            if chg < self.tolerance:
                break

            elbo_old = elbo_new

    def _update_beta(self, X): 
        V = X.shape[0]

        theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
        xauxsum = X / self._auxsum(theta)

        self.g = self.c0 / V + np.exp(self.Elogb) * np.dot(xauxsum, np.exp(theta).T)
        self.h = np.tile(self.c0 + np.sum(np.exp(theta), axis=1), (V, 1))

        self.Eb, self.Elogb = _compute_expectations(self.g, self.h)
        
    def _update_ql(self, X):
        for i, _ in enumerate(range(self.num_steps)):
            l_temp = np.zeros(self.l.shape)

            for k in range(self.K):
                Eb_k = self.Eb[:, k]
                Elogb_k = self.Elogb[:, k]
                
                theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
                theta_k = theta[k, :]
                xauxsum = X / self._auxsum(theta)
                
                grad = xauxsum * np.outer(np.exp(Elogb_k), np.exp(theta_k))
                grad -= np.outer(Eb_k, np.exp(theta_k))
                grad = np.sum((grad[:, :, np.newaxis] * self.u - self.l[k, :]), axis=(0,1))
                l_temp[k, :] = self.l[k, :] + self.step_size * grad

            self.l = l_temp

    def _update_qu(self, X):
        D = X.shape[1]

        for i, _ in enumerate(range(self.num_steps)):
            u_temp = np.zeros(self.u.shape)

            for d in range(D):
                theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
                theta_d = theta[:, d]
                xauxsum = X[:, d] / self._auxsum(theta_d)
                
                grad = xauxsum[:, np.newaxis] * np.exp(self.Elogb) * np.exp(theta_d[np.newaxis, :])
                grad -= self.Eb * np.exp(theta_d[np.newaxis, :])
                grad = np.sum((grad[:, :, np.newaxis] * self.l - self.u[d, :]), axis=(0,1))
                
                u_temp[d, :] = self.u[d, :] + self.step_size * grad

            self.u = u_temp

    def _update_alpha(self, X):
        a = np.log(np.sum(self.Eb[:, :, np.newaxis] * np.exp(np.dot(self.l, self.u.T)), axis=(0,1)) + 1e-100)
        self.alpha = np.log(np.sum(X, axis=0) + 1e-100) - a
        
    def _auxsum(self, theta):
        ''' 
        Sums the auxiliary parameter over the K dimension.
        
        Parameters
        ----------
        Elogb : array-like, shape (V, K)
        theta : array-like, shape (K, D)
        
        Returns
        -------
        auxsum : array-like, shape (V, D)
        '''  
        return np.dot(np.exp(self.Elogb), np.exp(theta))

    def _bound(self, X):
        V = X.shape[0]

        theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
        
        bound = np.sum(X * np.log(self._auxsum(theta) + + 1e-100) - np.dot(self.Eb, theta))
        bound += _gamma_term(self.c0, self.c0 / V, self.g, 
                             self.h, self.Eb, self.Elogb)
        bound += np.sum(np.log(norm.pdf(self.l, 0, self.sigma) + 1e-100))
        bound += np.sum(np.log(norm.pdf(self.u, 0, 1) + 1e-100))

        return bound
    
    
class StochasticCMF(CMF):
    def __init__(self, 
                 K=15, 
                 m=10,
                 num_steps=1,
                 step_size=1e-05,
                 n_epochs=1,
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
        self.m = m
        self.num_steps = num_steps
        self.step_size = step_size
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
        self.c0 = float(kwargs.get('c0', 0.1))
        self.sigma = float(kwargs.get('sigma', 1.0 / self.m))
        self.tau = float(kwargs.get('tau', 1.))
        self.kappa = float(kwargs.get('kappa', 0.6))

    def _init_ql(self, D):
        mean = np.zeros(self.m)
        cov = self.sigma * np.eye(self.m)
        self.l = np.random.multivariate_normal(mean, cov, self.K) + np.random.random() * self._scale

    def fit(self, X):
        V, D = X.shape

        self._scale = float(D) / self.minibatch_size

        if self.init == 'nmf':
            model = NMF(n_components=self.K, random_state=self.random_state)
            self.W = model.fit_transform(X.T) 
            self.H = model.components_ 
            model_m = NMF(n_components=self.m, random_state=self.random_state)
            self.W_m = model_m.fit_transform(self.W) 
            self.H_m = model_m.components_ 
            
        self._init_ql(D)
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
            X_shuffled = X[:, idxs]
            for (t, start) in enumerate(range(0, D, self.minibatch_size), 1):
                self.set_step_size(t=t)

                end = min(start + self.minibatch_size, D)
                minibatch = X_shuffled[:, start:end]
                
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
        self._stochastic_update_ql(X)
        self._stochastic_update_beta(X)

        return self
        
    def _stochastic_update_ql(self, X):
        l_temp = np.zeros(self.l.shape)

        for k in range(self.K):
            Eb_k = self.Eb[:, k]
            Elogb_k = self.Elogb[:, k]
            
            theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
            theta_k = theta[k, :]
            xauxsum = X / self._auxsum(theta)

            grad = xauxsum * np.outer(np.exp(Elogb_k), np.exp(theta_k))
            grad -= np.outer(Eb_k, np.exp(theta_k))
            grad = np.sum((grad[:, :, np.newaxis] * self.u - self.l[k, :]), axis=(0,1))
            
            G = self._calc_G(Eb_k, theta_k)
            l_temp[k, :] = self.l[k, :] + self.rho * np.dot(G, grad)

        self.l = l_temp

    def _stochastic_update_beta(self, X):
        V = X.shape[0]

        theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
        xauxsum = X / self._auxsum(theta)

        self.g = (1 - self.rho) * self.g + self.rho * (self.c0 / V + self._scale * \
                                                       np.exp(self.Elogb) * np.dot(xauxsum, np.exp(theta).T))
        self.h = (1 - self.rho) * self.h + self.rho * (np.tile(self.c0 + self._scale \
                                                               * np.sum(np.exp(theta), axis=1), (V, 1)))
        
        self.Eb, self.Elogb = _compute_expectations(self.g, self.h)

    def _calc_G(self, Eb_k, theta_k):
        '''
        Computes the preconditioning matrix G, 
        set to be the inverse negative Hessian.
        '''
        g1 = np.outer(Eb_k, np.exp(theta_k)) # V x D  
        g2 = np.einsum('ij,ik->ijk', self.u, self.u)
        g12 = np.sum((g1[:, :, np.newaxis, np.newaxis] * g2), axis=(0,1))

        G_inv = self.sigma**(-2) * np.eye(self.m) + self._scale * g12

        return np.linalg.inv(G_inv)

    def set_step_size(self, t=None):
        if t is not None:
            self.rho = (t + self.tau)**(-self.kappa)
        else:
            raise ValueError('Cannot set step size.')
        return self

    def _stochastic_bound(self, X):
        V = X.shape[0]

        theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
        
        bound = np.sum(X * np.log(self._auxsum(theta)) - np.dot(self.Eb, theta))
        bound += np.sum(np.log(norm.pdf(self.u, 0, 1)))

        bound *= self._scale

        bound += _gamma_term(self.c0, self.c0 / V, self.g, 
                             self.h, self.Eb, self.Elogb)
        bound += np.sum(np.log(norm.pdf(self.l, 0, self.sigma)))

        return bound
    