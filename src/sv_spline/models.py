import numpy as np
from abc import ABC, abstractmethod
from .filter import bootstrap_filter
from .transitions import mh_sv_basic
from .spline import Spline, K_gen


class SVModel(ABC):
    def __init__(self, y, x=None):
        self.y=y
        self.x=x

    @abstractmethod
    def is_valid(self, params):
        pass

    @abstractmethod
    def log_prior(self, params):
        pass

    @abstractmethod
    def transition(self, params):
        pass

    @abstractmethod
    def initial_distribution(self, params):
        pass

    def sigma(self,params):
        return np.sqrt(np.exp(params['log_sigma2']))

    def compute_y_centered(self,params):
        if self.x is not None:
            y_centered = self.y - (params['alpha'] + params['beta'] * self.x)
        else:
            y_centered = self.y - params['alpha']
        return y_centered

    def log_likelihood(self, params, M, rng, resampling="multinomial"):
        if not self.is_valid(params):
            return -np.inf
        y_centered = self.compute_y_centered(params)
        sigma = self.sigma(params)
        m_h = self.transition(params)
        initial_mean, initial_std = self.initial_distribution(params)
        log_likelihood, _, _ = bootstrap_filter(y_centered, sigma, initial_mean, initial_std, m_h, M, resampling=resampling, rng=rng)
        return log_likelihood

    def run_filter(self, params, M, rng, resampling="multinomial"):
        y_centered = self.compute_y_centered(params)
        sigma = self.sigma(params)
        m_h = self.transition(params)
        initial_mean, initial_std = self.initial_distribution(params)
        log_likelihood, increments, h = bootstrap_filter(y_centered, sigma, initial_mean, initial_std, m_h, M, resampling=resampling, rng=rng)
        return log_likelihood, increments, h

class BasicSV(SVModel):

    def is_valid(self,params):
        return abs(params['phi']) < 1 

    def log_prior(self,params):
        log_prior = 0
        if not self.is_valid(params):
            return -np.inf
        for param in params:
            log_prior += -(params[param]**2)/200
        return log_prior

    def transition(self,params):
        return mh_sv_basic(params['mu'], params['phi'])

    def initial_distribution(self,params):
        mean = params['mu']
        std = self.sigma(params)/np.sqrt(1-params['phi']**2)
        return mean, std

class SemiparametricSV(SVModel):
    def __init__(self, y, x, knots, lam, initial_mean, initial_std):
        super().__init__(y, x)
        self.knots = knots
        self.K = K_gen(knots)
        self.lam = lam
        self.initial_mean = initial_mean
        self.initial_std = initial_std

    def is_valid(self,params):
        spline = Spline(params['g'], self.knots)
        left_slope, right_slope = spline.get_slopes_ext()
        return abs(left_slope) <= 1 and abs(right_slope) <= 1

    def log_prior(self,params):
        log_prior = 0
        if not self.is_valid(params):
            return -np.inf
        for param in params:
            if param == 'g':
                log_prior += -0.5 * self.lam * params['g'].T @ self.K @ params['g']
            else:
                log_prior += -(params[param]**2)/200
        return log_prior

    def transition(self,params):
        return Spline(params['g'], self.knots)

    def initial_distribution(self,params):
        return self.initial_mean, self.initial_std

