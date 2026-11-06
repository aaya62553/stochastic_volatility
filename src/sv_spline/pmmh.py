from .models import SVModel
import numpy as np

def run_pmmh(model : SVModel, initial_params, blocks, n_warmup, n_iter, M, resampling = "multinomial", rng=None, verbose=False):
    if rng is None:
        rng = np.random.default_rng()
    
    params = initial_params.copy()
    acceptance_rates = {i: 0 for i in range(len(blocks))}
    params_history, ll_history = [], []

    log_prior = model.log_prior(params)
    if log_prior == -np.inf:
            raise ValueError("Initial parameters have zero prior probability. Please choose different initial parameters.")
    log_likelihood = model.log_likelihood(params, M, rng, resampling=resampling)
    if log_likelihood == -np.inf:
        raise ValueError("Initial parameters have zero likelihood. Please choose different initial parameters.")
    if verbose:
        print("Initial log probability:", log_prior + log_likelihood)
    
    for it in range(n_warmup + n_iter):
        for idx, block in enumerate(blocks):
            candidate = block.propose(params, rng)
            log_prior_candidate = model.log_prior(candidate)
            if log_prior_candidate == -np.inf:
                if it < n_warmup:
                    block.adapt(params, False)
                continue
            log_likelihood_candidate = model.log_likelihood(candidate, M, rng, resampling=resampling)
            log_ratio = (log_prior_candidate + log_likelihood_candidate) - (log_prior + log_likelihood) + block.log_q_ratio(params, candidate)
            if np.log(rng.random()) < log_ratio:
                log_prior = log_prior_candidate
                log_likelihood = log_likelihood_candidate
                params = candidate
                if it >= n_warmup:
                    acceptance_rates[idx] += 1/(n_iter)
                if it < n_warmup:
                    block.adapt(params, True)
            else:
                if it < n_warmup:
                    block.adapt(params, False)
        if verbose and (it + 1) % 500 == 0:
            print(f"Iteration {it + 1}/{n_warmup + n_iter}, Log probability: {log_prior + log_likelihood}")
        params_history.append(params)
        ll_history.append(log_likelihood)
    new_params_history = {}
    for param_hist in params_history:
        for key, value in param_hist.items():
            if key not in new_params_history:
                new_params_history[key] = []
            new_params_history[key].append(value)
    for key in new_params_history:
        new_params_history[key] = np.array(new_params_history[key])
    return {"params_history": new_params_history, 
            "ll_history": np.array(ll_history), 
            "acceptance_rates": acceptance_rates, 
            "n_warmup": n_warmup, "n_iter": n_iter,
            "model" : model}