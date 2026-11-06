import numpy as np
from .transitions import mh_dgp

def data_generating_process(alpha=0.01,beta=0.9, sigma_w=0.25, phi=0.7, T=1000, mu=-9.25, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    v = rng.normal(0,1,size=T)
    x = rng.uniform(-0.05, 0.05, size=T)
    w = rng.normal(0, 1, size=T)

    h = np.zeros(T)
    y = np.zeros(T)
    h[0] = rng.normal(mu, sigma_w/np.sqrt(1-phi**2))

    sigma_h = sigma_w/np.sqrt(1-phi**2)
    scale = 0.5 * sigma_h
    mh_simu = mh_dgp(phi, mu, scale)

    for t in range(T-1):
        y[t] = alpha+beta*x[t] + np.exp(h[t]/2)*v[t]
        h[t+1] = mh_simu(h[t])  + w[t]*sigma_w
    y[T-1] = alpha+beta*x[T-1] + np.exp(h[T-1]/2)*v[T-1]

    return y, h, x
