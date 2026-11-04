from scipy.stats import norm

def mh_sv_basic(mu, phi):
    return lambda h_prev: mu + phi * (h_prev - mu)

def mh_dpg(phi, center, scale):
    return lambda h_prev:  (-10 + 1.5 * norm.cdf((h_prev - center)/scale))*(1-phi) + phi*h_prev