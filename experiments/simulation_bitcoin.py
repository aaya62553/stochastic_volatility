import numpy as np
import pickle
import os

from sv_spline.data import load_data, split_data, get_y
from sv_spline.pmmh import run_pmmh
from sv_spline.proposals import RandomWalk, Composite, GProposal
from sv_spline.models import BasicSV, SemiparametricSV
from sv_spline.diagnostics import *
from sv_spline.spline import knots_gen

N_WARMUP = 1000
N_ITER = 2000
M = 2000
SAVE_RESULTS = True
SAVE_PATH = "results/simulation_bitcoin/"
SPLIT_DATE = "2021-01-01"
DATA_PATH = "data/btc-usd-max.csv"
SEED = 42
RETURN_COL = "Log Returns"


if __name__ == "__main__":

  rng = np.random.default_rng(SEED)

  # Load Bitcoin data
  df = load_data(DATA_PATH, scale=1.0)
  train_df, test_df = split_data(df, split_date=SPLIT_DATE)

  # Définition des blocs de propositions pour PMMH
  basic_steps={'alpha': 0.0003, 'mu': 0.05, 'phi': 0.01, 'log_sigma2': 0.05}
  random_walk_proposal = RandomWalk(keys=['alpha', 'log_sigma2', 'phi', 'mu'],
                                    scale=basic_steps)
  blocks = [random_walk_proposal]

  # Création du modèle SV de base
  y=get_y(df, y_col=RETURN_COL)
  y_train=get_y(train_df, y_col=RETURN_COL)
  y_test=get_y(test_df, y_col=RETURN_COL)
  basic_sv_model = BasicSV(y=y_train)

  alpha_init = y_train.mean()
  mu_init = 2 * np.log((alpha_init - y_train).std())

  # Exécution de l'algorithme PMMH
  print("Running PMMH for Basic SV model...")
  basic_pmmh_results = run_pmmh(model=basic_sv_model,
                              initial_params={'alpha': alpha_init, 
                                              'log_sigma2': -2.0, 'phi': 0.9, 
                                              'mu': mu_init},
                              blocks=blocks,
                              n_warmup=N_WARMUP,
                              n_iter=N_ITER,
                              M=M,
                              resampling="multinomial",
                              rng=rng,
                              verbose=True)


  post = posterior_params(basic_pmmh_results)
  h_hat = basic_sv_model.run_filter(post, M=M, rng=rng, resampling="multinomial")[2]


  # Semiparametric SV model
  # Initialisation des paramètres pour le modèle semiparamétrique
  K=6
  knots = knots_gen(a=np.quantile(h_hat,0.01), b=np.quantile(h_hat,0.99), n=K)
  g_init = post['mu']*(1-post['phi']) + post['phi']*knots
  initial_mean = post['mu']
  initial_std = basic_sv_model.sigma(post)/np.sqrt(1-post['phi']**2)
  alpha_init = post['alpha']
  log_sigma2_init = post['log_sigma2']
  lam = 0.01
  kernel_width = 0.01
  sigma_e = 0.05
  alpha_step = 0.0003
  log_sigma2_step = 0.05

  semi_sv_model = SemiparametricSV(y=y_train, knots=knots, lam=lam, 
                                  initial_mean=initial_mean, initial_std=initial_std)
  blocks_semi = [
      RandomWalk(keys=['alpha'], 
                scale={'alpha': alpha_step}),
      Composite([
        GProposal(semi_sv_model.K, kernel_width=kernel_width, sigma_e=sigma_e),
        RandomWalk(keys=['log_sigma2'], scale={'log_sigma2': log_sigma2_step}),
      ])
  ]

  print("Running PMMH for Semiparametric SV model...")
  semi_pmmh_results = run_pmmh(model=semi_sv_model,
                              initial_params={'alpha': alpha_init, 
                                              'log_sigma2': log_sigma2_init, 
                                              'g': g_init},
                              blocks=blocks_semi,
                              n_warmup=N_WARMUP,
                              n_iter=N_ITER,
                              M=M,
                              resampling="multinomial",
                              rng=rng,
                              verbose=True)

  # Save results
  if SAVE_RESULTS:
    data_to_save = {
        'df': df,
        'res_basic': basic_pmmh_results,
        'res_semi': semi_pmmh_results,
        'config': {
            'N_WARMUP': N_WARMUP,
            'N_ITER': N_ITER,
            'M': M,
            'SPLIT_DATE': SPLIT_DATE,
            'DATA_PATH': DATA_PATH,
            'SEED': SEED,
            'RETURN_COL': RETURN_COL,
            'K': K,
            'knots': knots,
            'lam': lam,
            'kernel_width': kernel_width,
            'sigma_e': sigma_e,
            'basic_steps': basic_steps,
            'semi_steps': {
                'alpha_step': alpha_step,
                'log_sigma2_step': log_sigma2_step
            }
        }
    }
    if os.path.exists(SAVE_PATH)== False:
        os.makedirs(SAVE_PATH)
    with open(SAVE_PATH + 'simulation_bitcoin_results.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)



  # Resultats Basic SV
  print("Summary Basic SV:", pmmh_summary(basic_pmmh_results))
  print("Acceptance rates Basic SV:", acceptance_rates_summary(basic_pmmh_results))

  plot_data(basic_pmmh_results, y=y_train, save_path=SAVE_PATH + "data_basic.png")
  plot_trace(basic_pmmh_results, save_path=SAVE_PATH + "trace_basic.png")

  print("Final parameter estimates Basic SV:",post)

  pred_outputs_basic = predictive_outputs(basic_pmmh_results, y_full=y, x_full=None, T_train=len(y_train), rng=rng)
  pit_basic_test=pred_outputs_basic['pit_test']
  print("Ljung-Box test for Basic SV model:")
  lb_test_basic, lb_test_sq_basic = ljungbox(pit_basic_test, lags=[5,10])
  print("Ljung-Box test for residuals of Basic SV model:")
  print(lb_test_basic)
  print("Ljung-Box test for squared residuals of Basic SV model:")
  print(lb_test_sq_basic)
  
  plot_pit(pit_basic_test, save_path=SAVE_PATH + "pit_basic.png")

  # Resultats Semiparametric SV
  print("Summary Semiparametric SV:", pmmh_summary(semi_pmmh_results))
  print("Acceptance rates Semiparametric SV:", acceptance_rates_summary(semi_pmmh_results))

  plot_data(semi_pmmh_results, y=y_train, save_path=SAVE_PATH + "data_semi.png")
  plot_trace(semi_pmmh_results, save_path=SAVE_PATH + "trace_semi.png")
  
  print("Final parameter estimates Semiparametric SV:", posterior_params(semi_pmmh_results))
  
  pred_outputs_semi = predictive_outputs(semi_pmmh_results, y_full=y, x_full=None, T_train=len(y_train), rng=rng)
  pit_semi_test=pred_outputs_semi['pit_test']

  print("Ljung-Box test for Semiparametric SV model:")
  lb_test_semi, lb_test_sq_semi = ljungbox(pit_semi_test, lags=[5,10])
  print("Ljung-Box test for residuals of Semiparametric SV model:")
  print(lb_test_semi)
  print("Ljung-Box test for squared residuals of Semiparametric SV model:")
  print(lb_test_sq_semi)
  
  plot_pit(pit_semi_test, save_path=SAVE_PATH + "pit_semi.png")

  compare_plot_mh(basic_pmmh_results, semi_pmmh_results, mh_true=None, rng=rng, save_path=SAVE_PATH + "compare_mh.png")
  plot_mh_band(semi_pmmh_results, rng=rng, save_path=SAVE_PATH + "mh_band_semi.png")
  comparaison_oos = compare_oos(basic_pmmh_results, semi_pmmh_results, y_full=y, x_full=None, T_train=len(y_train), seed=42)
  print("Out-of-sample comparison:")
  print(comparaison_oos)
  comparaison_oos.to_csv(SAVE_PATH + "oos_comparison.csv", index=False)




 