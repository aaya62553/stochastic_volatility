import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression

from sv_spline.simulation import data_generating_process
from sv_spline.pmmh import run_pmmh
from sv_spline.proposals import RandomWalk, Composite, GProposal
from sv_spline.models import BasicSV, SemiparametricSV
from sv_spline.diagnostics import *
from sv_spline.spline import knots_gen

N_WARMUP = 500
N_ITER = 1000
M = 1000
SAVE_RESULTS = True
SAVE_PATH = "results/simulation_single/"


if __name__ == "__main__":

  rng = np.random.default_rng(42)

  # Génération des données avec la fonction data_generating_process
  true_alpha = 0.01
  true_beta = 0.9
  true_sigma_w = 0.25
  true_phi = 0.7
  true_mu = -9.25
  y, h, x, mh_simu = data_generating_process(alpha=true_alpha, 
                                    beta=true_beta, 
                                    sigma_w=true_sigma_w, 
                                    phi=true_phi, 
                                    mu=true_mu, 
                                    T=1000, rng=rng)

  # Définition des blocs de propositions pour PMMH
  random_walk_proposal = RandomWalk(keys=['alpha', 'beta', 'log_sigma2', 'phi', 'mu'],
                                    scale={'alpha': 0.0003, 'beta': 0.01, 'mu': 0.05, 
                                            'phi': 0.01, 'log_sigma2': 0.05, 
                                            })
  blocks = [random_walk_proposal]

  # Création du modèle SV de base
  basic_sv_model = BasicSV(y=y, x=x)

  reg = LinearRegression().fit(x.reshape(-1, 1), y)
  alpha_init = reg.intercept_
  beta_init = reg.coef_[0]
  mu_init = 2 * np.log((x*beta_init + alpha_init - y).std())

  # Exécution de l'algorithme PMMH
  print("Running PMMH for Basic SV model...")
  basic_pmmh_results = run_pmmh(model=basic_sv_model,
                              initial_params={'alpha': alpha_init, 'beta': beta_init, 
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
  h_hat = basic_sv_model.run_filter(post, M=1000, rng=rng, resampling="multinomial")[2]


  # Semiparametric SV model
  # Initialisation des paramètres pour le modèle semiparamétrique
  K=6
  knots = knots_gen(a=np.quantile(h_hat,0.01), b=np.quantile(h_hat,0.99), n=K)
  g_init = post['mu']*(1-post['phi']) + post['phi']*knots
  initial_mean = post['mu']
  initial_std = basic_sv_model.sigma(post)/np.sqrt(1-post['phi']**2)
  alpha_init = post['alpha']
  beta_init = post['beta']
  log_sigma2_init = post['log_sigma2']

  semi_sv_model = SemiparametricSV(y=y, x=x, knots=knots, lam=0.01, 
                                  initial_mean=initial_mean, initial_std=initial_std)
  blocks_semi = [
      RandomWalk(keys=['alpha', 'beta'], 
                scale={'alpha': 0.0003, 'beta': 0.01}),
      Composite([
        GProposal(semi_sv_model.K, kernel_width=0.01, sigma_e=0.05),
        RandomWalk(keys=['log_sigma2'], scale={'log_sigma2': 0.05}),
      ])
  ]

  print("Running PMMH for Semiparametric SV model...")
  semi_pmmh_results = run_pmmh(model=semi_sv_model,
                              initial_params={'alpha': alpha_init, 'beta': beta_init, 
                                              'log_sigma2': log_sigma2_init, 'g': g_init},
                              blocks=blocks_semi,
                              n_warmup=N_WARMUP,
                              n_iter=N_ITER,
                              M=M,
                              resampling="multinomial",
                              rng=rng,
                              verbose=True)

 # Save results
  if SAVE_RESULTS:
    if os.path.exists(SAVE_PATH) == False:
        os.makedirs(SAVE_PATH)

    data_to_save = {
        'y': y,
        'h': h,
        'x': x,
        'res_basic': basic_pmmh_results,
        'res_semi': semi_pmmh_results,}

    with open(SAVE_PATH + 'simulation_single_results.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)



  # Resultats Basic SV
  print("Summary Basic SV:", pmmh_summary(basic_pmmh_results))
  print("Acceptance rates Basic SV:", acceptance_rates_summary(basic_pmmh_results))

  plot_data(basic_pmmh_results, y=y, x=x, save_path=SAVE_PATH + "data_basic.png")
  plot_trace(basic_pmmh_results, save_path=SAVE_PATH + "trace_basic.png")

  print("Final parameter estimates Basic SV:",post)

  print("IMSE Basic SV:",imse(basic_pmmh_results, mh_true=mh_simu, h_true=h))
  pit_basic = compute_pit(basic_pmmh_results)
  lb_test_basic, lb_test_sq_basic = ljungbox(pit_basic, lags=[5,10])
  print("Ljung-Box test for residuals of Basic SV model:")
  print(lb_test_basic)
  print("Ljung-Box test for squared residuals of Basic SV model:")
  print(lb_test_sq_basic)
  
  plot_pit(pit_basic, save_path=SAVE_PATH + "pit_basic.png")

  # Resultats Semiparametric SV
  print("Summary Semiparametric SV:", pmmh_summary(semi_pmmh_results))
  print("Acceptance rates Semiparametric SV:", acceptance_rates_summary(semi_pmmh_results))
  plot_data(semi_pmmh_results, y=y, x=x, save_path=SAVE_PATH + "data_semi.png")
  plot_trace(semi_pmmh_results, save_path=SAVE_PATH + "trace_semi.png")
  
  print("Final parameter estimates Semiparametric SV:", posterior_params(semi_pmmh_results))
  
  print("IMSE Semiparametric SV:", imse(semi_pmmh_results, mh_true=mh_simu, h_true=h))
  pit_semi = compute_pit(semi_pmmh_results)

  lb_test_semi, lb_test_sq_semi = ljungbox(pit_semi, lags=[5,10])
  print("Ljung-Box test for residuals of Semiparametric SV model:")
  print(lb_test_semi)
  print("Ljung-Box test for squared residuals of Semiparametric SV model:")
  print(lb_test_sq_semi)
  
  plot_pit(pit_semi, save_path=SAVE_PATH + "pit_semi.png")

  compare_plot_mh(basic_pmmh_results, semi_pmmh_results, mh_true=mh_simu, rng=rng, save_path=SAVE_PATH + "compare_mh.png")
  plot_mh_band(semi_pmmh_results, rng=rng, save_path=SAVE_PATH + "mh_band_semi.png")
  comparaison_oos = compare_oos(basic_pmmh_results, semi_pmmh_results, y_full=y, x_full=x, T_train=800, seed=42)
  print("Out-of-Sample Comparison:")
  print(comparaison_oos)
  comparaison_oos.to_csv(SAVE_PATH + "oos_comparison.csv", index=False)

 