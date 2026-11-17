import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.stats import norm, kstest
from .spline import Spline
from statsmodels.stats.diagnostic import acorr_ljungbox

def plot_data(pmmh_results,y,x=None,true_h=None, save_path=None):
    model = pmmh_results["model"]
    params = posterior_params(pmmh_results)
    _, _, h, _ = model.run_filter(params, M=1000, rng=np.random.default_rng(), resampling="multinomial", y=y, x=x)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(y, label='y', color='blue')
    plt.title('Observed Data y')
    plt.xlabel('Time')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(h, label='h', color='orange')
    if true_h is not None:
        plt.plot(true_h, label='True h', color='red', linestyle='--')
    plt.title('Estimated Latent Volatility h')
    plt.xlabel('Time')
    plt.ylabel('h')
    plt.legend()
    plt.title('Latent Volatility h')
    

    if x is not None:
        plt.subplot(3, 1, 3)
        plt.plot(x, label='x', color='green')
        plt.title('Exogenous Variable x')
        plt.xlabel('Time')
        plt.ylabel('x')
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

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

def plot_trace(pmmh_results, save_path=None):
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
        plt.subplot(rows, cols, j + 1)
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
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_mh_band(pmmh_results, rng=None, save_path=None):
    """
    Uniquement pour le modèle semiparamétrique, trace de m_h et intervalle de 
    crédibilité à 95% pour m_h
    """
    model = pmmh_results["model"]
    params = posterior_params(pmmh_results)
    n_warmup = pmmh_results["n_warmup"]
    if rng is None:
        rng = np.random.default_rng()
    _, _, h, _ = model.run_filter(params, M=1000, rng=rng, resampling="multinomial") 
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
    if save_path:
        plt.savefig(save_path)
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




def compare_plot_mh(pmmh_results_basic, pmmh_results_semi, mh_true=None, rng=None, save_path=None):
    if rng is None:
        rng = np.random.default_rng()
    model_basic = pmmh_results_basic["model"]
    model_semi = pmmh_results_semi["model"]
    params_basic = posterior_params(pmmh_results_basic)
    params_semi = posterior_params(pmmh_results_semi)
    _, _, h_basic, _ = model_basic.run_filter(params_basic, M=1000, rng=rng, resampling="multinomial") 
    _, _, h_semi, _ = model_semi.run_filter(params_semi, M=1000, rng=rng, resampling="multinomial") 
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
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_pit(pmmh_results, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    model = pmmh_results["model"]
    params = posterior_params(pmmh_results)
    _, _, _, pit = model.run_filter(params, M=1000, rng=rng, resampling="multinomial") 
    return pit

def plot_pit(pit, save_path=None):
    """
    Plot the histogram and Q-Q plot of PIT values for a given PMMH result.
    We compute the probability that the model predicts a return less than the 
    observed return, given the estimated volatility. If the model is well-calibrated,
    the PIT values should be uniformly distributed.
    """
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(pit, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Histogram of PIT values')
    plt.xlabel('PIT value')
    plt.ylabel('Density')

    plt.subplot(1, 2, 2)
    probplot(pit, dist="uniform", plot=plt)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid()
    plt.title('Q-Q plot of PIT values vs Uniform[0,1]')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def ljungbox(pit, lags=[5,10]):
    residual = norm.ppf(pit)
    lb_test = acorr_ljungbox(residual, lags=lags, return_df=True)
    lb_test_sq = acorr_ljungbox(residual**2, lags=lags, return_df=True)
    return lb_test, lb_test_sq

def predictive_outputs(pmmh_results, y_full, x_full, T_train, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    model = pmmh_results["model"]
    params = posterior_params(pmmh_results)
    _, increments, h, pit = model.run_filter(params, M=1000, rng=rng, resampling="multinomial", y=y_full, x=x_full)
    return {
        "pit_train": pit[:T_train],
        "pit_test": pit[T_train:],
        "h":h,
        "log_score_test": np.sum(increments[T_train:]),
    }

def compare_oos(pmmh_results_basic, pmmh_results_semi, y_full, x_full, T_train, seed=0):
    outputs_basic = predictive_outputs(pmmh_results_basic, y_full, x_full, T_train, rng=np.random.default_rng(seed))
    outputs_semi = predictive_outputs(pmmh_results_semi, y_full, x_full, T_train, rng=np.random.default_rng(seed))
    ljungbox_basic, ljungbox_sq_basic = ljungbox(outputs_basic["pit_test"], lags=[5,10])
    ljungbox_semi, ljungbox_sq_semi = ljungbox(outputs_semi["pit_test"], lags=[5,10])

    results = pd.DataFrame({
        "Model": ["Basic SV", "Semiparametric SV"],
        "Log Score Test": [outputs_basic["log_score_test"], outputs_semi["log_score_test"]],
        "KS Test p-value": [kstest(outputs_basic["pit_test"], 'uniform').pvalue, kstest(outputs_semi["pit_test"], 'uniform').pvalue],
        "Ljung-Box Test (Residuals) p-value": [ljungbox_basic["lb_pvalue"].values[-1], ljungbox_semi["lb_pvalue"].values[-1]],
        "Ljung-Box Test (Squared Residuals) p-value": [ljungbox_sq_basic["lb_pvalue"].values[-1], ljungbox_sq_semi["lb_pvalue"].values[-1]],
    })
    return results