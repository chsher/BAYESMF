# adapted from @dawenl stochastic_PMF

import numpy as np
from scipy import special
from sklearn.base import BaseEstimator, TransformerMixin


ITER_STMT = 'Iter: {0:d}, Bound: {1:.2f}, Change: {2:.5f}'
EPOCH_STMT = 'Epoch: {0:d}'
MINIBATCH_STMT = 'Minibatch: {0:d}'


def _compute_expectations(a, b):
    '''
    Computes the expectation of X and of log(X),
    where X ~ Gamma(a, b) with shape parameter
    a and rate parameter b.
    '''
    Ex = a / b
    Elogx = special.psi(a) - np.log(b) 
    return Ex, Elogx


def _gamma_term(a, b, shape, rate, Ex, Elogx):
    '''
    Computes E_q[p(X)] - E_q[q(X)] where:
        p(X) = Gamma(a, b)
        q(X) = Gamma(shape, rate)
    excluding constant terms wrt X, shape, rate.
    '''
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  special.gammaln(shape) - shape * np.log(rate)) 


class BMF(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 K=15, 
                 max_iters=100, 
                 tolerance=0.0005, 
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
        self.a0 = float(kwargs.get('a0', 0.1))
        self.b0 = float(kwargs.get('b0', 0.1))
        self.c0 = float(kwargs.get('c0', 0.1))

    def _init_qbeta(self, V):
        self.g = np.random.gamma(self.smoothness, 
                                 scale = 1.0 / self.smoothness, 
                                 size=(V, self.K))
        self.h = np.random.gamma(self.smoothness, 
                                 scale = 1.0 / self.smoothness, 
                                 size=(V, self.K))
        self.Eb, self.Elogb = _compute_expectations(self.g, self.h)

    def _init_qtheta(self, D):
        self.a = np.random.gamma(self.smoothness, 
                                 scale = 1.0 / self.smoothness, 
                                 size=(self.K, D))
        self.b = np.random.gamma(self.smoothness, 
                                 scale = 1.0 / self.smoothness, 
                                 size=(self.K, D))
        self.Et, self.Elogt = _compute_expectations(self.a, self.b)

    def fit(self, X):
        V, D = X.shape
        self._init_qbeta(V)
        self._init_qtheta(D)
        self._update(X)
        return self

    def transform(self, X, attr='Et'):
        if not hasattr(self, 'Eb'):
            raise ValueError('No beta initialized.')

        V, D = X.shape
        if not self.Eb.shape[0] == V:
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
            #chg = np.exp(logsumexp(elbo_new - elbo_old) - logsumexp(elbo_old))
            
            if self.verbose and i % 10 == 0:
                print(ITER_STMT.format(i, elbo_new, chg))

            if chg < self.tolerance:
                break

            elbo_old = elbo_new

    def _update_beta(self, X): 
        V = X.shape[0]
        xauxsum = X / self._auxsum()
        # (V x K) * (V x D) (K x D)^T
        self.g = self.c0 / V + np.exp(self.Elogb) * np.dot(xauxsum, np.exp(self.Elogt).T)
        #self.h = self.c0 + np.sum(self.Et, axis=1)
        self.h = np.expand_dims(self.c0 + np.sum(self.Et, axis=1), axis=0)
        self.Eb, self.Elogb = _compute_expectations(self.g, self.h)
        
    def _update_theta(self, X):
        xauxsum = X / self._auxsum()
        # (K x D) * (V x K)^T (V x D)
        self.a = self.a0 + np.exp(self.Elogt) * np.dot(np.exp(self.Elogb).T, xauxsum)
        #self.b = self.b0 + np.sum(self.Eb, axis=0)
        self.b = np.expand_dims(self.b0 + np.sum(self.Eb, axis=0), axis=1)
        self.Et, self.Elogt = _compute_expectations(self.a, self.b)

    def _auxsum(self):
        ''' 
        Sums the auxiliary parameter over the K dimension.
        Elogb: V x K
        Elogt: K x D
        Returns: V x D
        '''
        return np.dot(np.exp(self.Elogb), np.exp(self.Elogt))

    def _bound(self, X):
        V, D = X.shape
        bound = np.sum(X * np.log(self._auxsum()) - np.dot(self.Eb, self.Et))
        bound += _gamma_term(self.c0, self.c0 / V, self.g, 
                             self.h, self.Eb, self.Elogb)
        bound += _gamma_term(self.a0, self.b0, self.a, 
                             self.b, self.Et, self.Elogt)
        return bound


class StochasticBMF(BMF):
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

        if type(self.random_state) is int:
            np.random.seed(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        self.a0 = float(kwargs.get('a0', 0.1))
        self.b0 = float(kwargs.get('b0', 0.1))
        self.c0 = float(kwargs.get('c0', 0.1))
        self.tau = float(kwargs.get('tau', 1.))
        self.kappa = float(kwargs.get('kappa', 0.6))

    def fit(self, X):
        V, D = X.shape

        self._scale = float(D) / self.minibatch_size
        self._init_qbeta(V)
        self.bound = []

        for e in range(self.n_epochs):
            if self.verbose:
                print(EPOCH_STMT.format(e + 1))

            idxs = np.arange(D)
            if self.shuffle:
                np.random.shuffle(idxs)

            X_shuffled = X[:, idxs]
            for (t, start) in enumerate(range(0, D, self.minibatch_size), 1):
                if self.verbose:
                    print(MINIBATCH_STMT.format(t))
                self.set_step_size(t=t)

                end = min(start + self.minibatch_size, D)
                minibatch = X_shuffled[:, start:end]
                self.partial_fit(minibatch)

                self.bound.append(self._stochastic_bound(minibatch))
        return self

    def partial_fit(self, X):
        self.transform(X)
        V = X.shape[0]
        xauxsum = X / self._auxsum()
        # (V x K) * (V x D) (K x D)^T
        self.g = (1 - self.rho) * self.g + self.rho * (self.c0 / V + self._scale * np.exp(self.Elogb) \
                                                       * np.dot(xauxsum, np.exp(self.Elogt).T))
        self.h = (1 - self.rho) * self.h + self.rho * (np.expand_dims(self.c0 + self._scale * \
                                                                      np.sum(self.Et, axis=1), axis=0))
        self.Eb, self.Elogb = _compute_expectations(self.g, self.h)
        return self
        
    def set_step_size(self, t=None):
        if t is not None:
            self.rho = (t + self.tau)**(-self.kappa)
        else:
            raise ValueError('Cannot set step size.')
        return self

    def _stochastic_bound(self, X):
        V, D = X.shape
        bound = np.sum(X * np.log(self._auxsum()) - np.dot(self.Eb, self.Et))
        bound += _gamma_term(self.a0, self.b0, self.a, 
                            self.b, self.Et, self.Elogt)
        bound *= self._scale
        bound += _gamma_term(self.c0, self.c0 / V, self.g, 
                            self.h, self.Eb, self.Elogb)
        return bound
