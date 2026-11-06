import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .spline import Spline

def posterior_params(pmmh_results):
    params_history = pmmh_results["params_history"]
    n_warmup = pmmh_results["n_warmup"]
    posterior_params = {}
    for key in params_history:
        posterior_params[key] = np.mean(params_history[key][n_warmup:], axis=0)
    return posterior_params


def pmmh_summary(pmmh_results):
    params_history = pmmh_results["params_history"]
    n_warmup = pmmh_results["n_warmup"]
    summary = {"Parameter": [], "Mean": [], "Std": [], "Quantile 2.5%": [], "Quantile 97.5%": []}
    for key in params_history:
        if key == 'g':
            continue
        param_samples = params_history[key][n_warmup:]
        summary["Parameter"].append(key)
        summary["Mean"].append(param_samples.mean())
        summary["Std"].append(param_samples.std())
        summary["Quantile 2.5%"].append(np.quantile(param_samples, 0.025))
        summary["Quantile 97.5%"].append(np.quantile(param_samples, 0.975))
    return pd.DataFrame(summary)

def acceptance_rates_summary(pmmh_results):
    acceptance_rates = pmmh_results["acceptance_rates"]
    summary = {"Block": [], "Acceptance Rate": []}
    for block, rate in acceptance_rates.items():
        summary["Block"].append(block)
        summary["Acceptance Rate"].append(rate)
    return pd.DataFrame(summary)

def plot_trace(pmmh_results):
    parameters_history = pmmh_results["params_history"]
    likelihood_history = pmmh_results["ll_history"]
    n = len(parameters_history) if 'g' in parameters_history else len(parameters_history) + 1
    plt.figure(figsize=(12, 10))
    cols = np.ceil(np.sqrt(n)).astype(int)
    rows = np.ceil(n / cols).astype(int)
    j=0
    for i, param in enumerate(parameters_history):
        if param == 'g':
            continue
        plt.subplot(rows, cols, i + 1)
        plt.plot(parameters_history[param], label=param)
        plt.title(f'Trace of {param}')
        plt.xlabel('Iteration')
        plt.ylabel(param)
        j+=1
    plt.subplot(rows, cols, j + 1)
    plt.plot(likelihood_history, label='Log Likelihood')
    plt.title('Trace of Log Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()

def plot_mh_band(pmmh_results, rng=None):
    """
    Uniquement pour le modèle semiparamétrique, trace de m_h et intervalle de 
    crédibilité à 95% pour m_h
    """
    model = pmmh_results["model"]
    params = posterior_params(pmmh_results)
    n_warmup = pmmh_results["n_warmup"]
    if rng is None:
        rng = np.random.default_rng()
    _, _, h = model.run_filter(params, M=1000, rng=rng, resampling="multinomial") 
    grid = np.linspace(np.quantile(h, 0.01), np.quantile(h, 0.99), 100)
    g_posterior = pmmh_results["params_history"]["g"][n_warmup:]
    matrix = np.array([Spline(g, model.knots)(grid) for g in g_posterior])
    g_mean = np.mean(matrix, axis=0)
    quantiles = np.quantile(matrix, [0.025, 0.975], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(grid, g_mean, label='Posterior Mean of m_h', color='blue')
    plt.fill_between(grid, quantiles[0], quantiles[1], color='blue', alpha=0.2, label='95% Credible Interval')
    plt.title('Posterior Mean and 95% Credible Interval of m_h')
    plt.xlabel('h')
    plt.ylabel('m_h(h)')
    plt.legend()
    plt.show()

def imse(pmmh_results, mh_true, h_true):
    model = pmmh_results["model"]
    params = posterior_params(pmmh_results)
    mh_hat = model.transition(params)
    h_low,h_high = np.quantile(h_true, [0.05, 0.95])
    bound = int((h_high - h_low)/0.01)
    res=0
    for i in range(bound):
        res+= (mh_true(h_low + i * 0.01) - mh_hat(h_low + i * 0.01))**2 
    return res*0.01



def compare_plot_mh(pmmh_results_basic, pmmh_results_semi, mh_true=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    model_basic = pmmh_results_basic["model"]
    model_semi = pmmh_results_semi["model"]
    params_basic = posterior_params(pmmh_results_basic)
    params_semi = posterior_params(pmmh_results_semi)
    _, _, h_basic = model_basic.run_filter(params_basic, M=1000, rng=rng, resampling="multinomial") 
    _, _, h_semi = model_semi.run_filter(params_semi, M=1000, rng=rng, resampling="multinomial") 
    mh_hat_basic = model_basic.transition(params_basic)
    mh_hat_semi = model_semi.transition(params_semi)
    grid = np.linspace(min(np.quantile(h_basic, 0.01), np.quantile(h_semi, 0.01)), max(np.quantile(h_basic, 0.99), np.quantile(h_semi, 0.99)), 100)
    plt.figure(figsize=(10, 6))
    plt.plot(grid, mh_hat_basic(grid), label='Estimated m_h (Basic SV)', color='blue')
    plt.plot(grid, mh_hat_semi(grid), label='Estimated m_h (Semiparametric SV)', color='orange')
    if mh_true is not None:
        plt.plot(grid, mh_true(grid), label='True m_h', color='green', linestyle='--')
    plt.title('Comparison of Estimated m_h Functions')
    plt.xlabel('h')
    plt.ylabel('m_h(h)')
    plt.legend()
    plt.show()