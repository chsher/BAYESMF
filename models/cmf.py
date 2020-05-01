import numpy as np
from scipy import special
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin


ITER_STMT = 'Iter: {0:d}, Bound: {1:.2f}, Change: {2:.5f}'


def _compute_expectations(a, b):
    Ex = a / b
    Elogx = special.psi(a) - np.log(b) 
    return Ex, Elogx

def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  special.gammaln(shape) - shape * np.log(rate)) 
                  

class CMF(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 K=15, 
                 m=10,
                 num_steps=10,
                 step_size=0.001,
                 max_iters=100, 
                 tolerance=0.0005, 
                 smoothness=100, 
                 random_state=22690, 
                 verbose=False,
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

        if type(self.random_state) is int:
            np.random.seed(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        self.c0 = float(kwargs.get('c0', 0.1))
        self.sigma = float(kwargs.get('sigma', 1.0))

    def _init_qbeta(self, V):
        self.g = np.random.gamma(self.smoothness, 
                                 scale = 1.0 / self.smoothness, 
                                 size=(V, self.K))
        self.h = np.random.gamma(self.smoothness, 
                                 scale = 1.0 / self.smoothness, 
                                 size=(V, self.K))
        self.Eb, self.Elogb = _compute_expectations(self.g, self.h)

    # K x m
    def _init_ql(self):
        mean = np.zeros(self.m)
        cov = self.sigma * np.eye(self.m)
        self.l = np.random.multivariate_normal(mean, cov, self.K)

    # D x m
    def _init_qu(self, D):
        mean = np.zeros(self.m)
        cov = np.eye(self.m)
        self.u = np.random.multivariate_normal(mean, cov, D)

    def _init_alpha(self, D):
        mean = 0
        sd = 1
        self.alpha = np.random.normal(mean, sd, D)

    def fit(self, X):
        V, D = X.shape
        self._init_qbeta(V)
        self._init_ql()
        self._init_qu(D)
        self._init_alpha(D)
        self._update(X)
        return self

    def transform(self, X, attr='l'):
        if not hasattr(self, 'Eb'):
            raise ValueError('No beta initialized.')

        V, D = X.shape
        if not self.Eb.shape[0] == V:
            raise ValueError('Feature dim mismatch.')

        self._init_ql()
        self._init_qu(D)
        self._init_alpha(D)
        self._update(X, update_beta=False)

        return getattr(self, attr)

    def _update(self, X, update_beta=True):
        elbo_old = -np.inf
        for i in range(self.max_iters):
            self._update_ql(X)
            self._update_qu(X)
            self._update_alpha(X)

            if update_beta:
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

        self.g = self.c0 / V + np.exp(self.Elogb) * np.dot(xauxsum, theta.T)
        self.h = np.expand_dims(self.c0 + np.sum(np.exp(theta), axis=1), axis=0)

        self.Eb, self.Elogb = _compute_expectations(self.g, self.h)
        
    def _update_ql(self, X):
        for k in range(self.K):
            Eb_k = self.Eb[:, k]
            Elogb_k = self.Elogb[:, k]
            
            elbo_old = -1e08
            for i, _ in enumerate(range(self.num_steps)):
                theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
                theta_k = theta[k, :]
                xauxsum = X / self._auxsum(theta)
            
                grad = xauxsum * np.outer(np.exp(Elogb_k), np.exp(theta_k))
                grad -= np.outer(Eb_k, np.exp(theta_k))
                grad = np.dot(np.sum(grad, axis=0), self.u) - self.sigma**(-2) * self.l[k, :]
    
                #l_old = self.l
                self.l[k, :] = self.l[k, :] - self.step_size * grad
                #elbo_new = self._bound(X)
                #if elbo_new < elbo_old:
                #    print('bad elbo:', elbo_new)
                #    self.l = l_old
                #    break
                #else:
                #    elbo_old = elbo_new

    def _update_qu(self, X):
        D = X.shape[1]

        for d in range(D):
            for i, _ in enumerate(range(self.num_steps)):
                theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
                theta_d = theta[:, d]
                xauxsum = X[:, d] / self._auxsum(theta_d)
                
                grad = xauxsum[:, np.newaxis] * np.exp(self.Elogb) * np.exp(theta_d[np.newaxis, :])
                grad -= self.Eb * np.exp(theta_d[np.newaxis, :])
                grad = np.dot(np.sum(grad, axis=0), self.l) - self.u[d, :]
                self.u[d, :] = self.u[d, :] - self.step_size * grad

    def _update_alpha(self, X):
        self.alpha = np.log(np.sum(X, axis=0)) - np.log(np.sum(np.dot(self.Eb, np.exp(np.dot(self.l, self.u.T))), axis=0))

    def _auxsum(self, theta):
        ''' 
        Sums the auxiliary parameter over the K dimension.
        Elogb: V x K
        theta: K x D
        auxsum: V x D
        '''
        return np.dot(np.exp(self.Elogb), theta)

    def _bound(self, X):
        V, D = X.shape

        theta = self.alpha[np.newaxis, :] + np.dot(self.l, self.u.T)
        
        bound = np.sum(X * np.log(self._auxsum(theta)) - np.dot(self.Eb, theta))
        bound += _gamma_term(self.c0, self.c0 / V, self.g, 
                             self.h, self.Eb, self.Elogb)
        bound += np.sum(np.log(norm.pdf(self.l, 0, self.sigma)))
        bound += np.sum(np.log(norm.pdf(self.u, 0, 1)))

        return bound
