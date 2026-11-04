import numpy as np

def resample_particles_multinomial(particles, weights, rng = np.random.default_rng()):
    return rng.choice(particles, size=len(particles), p=weights)

def resample_particles_systematic(particles, weights, rng = np.random.default_rng()):
    N = len(particles)
    positions = (rng.random() + np.arange(N)) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off errors
    indexes = np.searchsorted(cumulative_sum, positions, side='right', sorter=None)

    return particles[indexes]

def bootstrap_filter(y_centered, sigma, initial_mean, initial_std, m_h, M, resampling="multinomial", rng = None):
    if rng is None:
        rng = np.random.default_rng()
    T = len(y_centered)
    increments = np.zeros(T)
    h=np.zeros(T)
    particles = rng.normal(initial_mean, initial_std, size=M)
    for t in range(T):
        particles = m_h(particles) + rng.normal(0, sigma, size=M)
        weights = -0.5*np.log(2*np.pi) -0.5*particles - (y_centered[t])**2 / (2*np.exp(particles))
        max_weight = np.max(weights)
        exp_weights = np.exp(weights - max_weight)
        increments[t] = np.log(np.sum(exp_weights)) + max_weight - np.log(M)
        if max_weight == -np.inf or np.isnan(max_weight):
            print(f"Warning: Max weight is -inf or NaN at time {t}")
            return -np.inf, increments, h
        weights = exp_weights/np.sum(exp_weights)
        h[t] = np.sum(particles * weights)
        if resampling == "multinomial":
            particles = resample_particles_multinomial(particles, weights, rng=rng)
        elif resampling == "systematic":
            particles = resample_particles_systematic(particles, weights, rng=rng)
        else:
            raise ValueError("Invalid resampling method. Choose 'multinomial' or 'systematic'.")
    log_likelihood = np.sum(increments)
    return log_likelihood, increments, h