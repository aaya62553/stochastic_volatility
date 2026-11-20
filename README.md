# Implementation of a semiparametric Stochastic Volatility model

This repository tries to compare a basic stochastic volatility (SV) model and a semiparametric SV model using sequential Monte Carlo algorithm and particle Markov chain Monte carlo methods. We implemented the model proposed by [Wang & Lou (2023)](https://doi.org/10.1016/j.econmod.2023.106287) (_Sequential Bayesian analysis for semiparametric stochastic volatility model with applications_) and tried to add some extensions.

## 1. Context

### 1.1 Data Generating Process

To test our model, we want to generate data so that we can observe the true volatility and compare with the volatility we predict.

We can model a return equation :
$y_t = \alpha + \beta x_t + \exp(h_t/2) v_t$

with :

- $\alpha$ : mean of returns (ex : 0.01, ie 1% of daily return)
- $x_t \sim \mathcal{U}(-0.05,0.05)$
- $\beta$ : Coefficient of sensitivity to $x_t$ (ex : 0.9). Increasing this coeff increase the variance of the observed return y.
- $\exp(h_t/2) v_t$ : Stochastic volatility component with $v_t \sim \mathcal{N}(0,1)$

We can then model the volatility $h_t$ using an AR(1) process :

$h_{t+1} = \mu (1-\phi) + \phi h_t + \sigma w_t$

where :

- $\mu$ : long term mean
- $\phi$ : return to mean speed, if $\phi \rightarrow 0$, we return very fast to the mean, if $\phi \rightarrow 1$ we return to the mean more slowly
- $w_t \sim \mathcal{N}(0,1)$

In our case, we replace $\mu = \mu(h_t) = -10 +1.5 cdf(.)$ => The mean depend on the current volatility state.

- $cdf$ if the CDF of the gaussian law.
- Quand la vol est basse (ie $h_t \to -\infty$) cdf tend vers 0 donc la moyenne cible est $\mu(h_t) = -10$
- Quand la vol est haute (ie $h_t \to \infty$) cdf tend vers 1 donc la moyenne cible est $\mu(h_t) = -10+1.5 = -8.5$
- Au numérateur de la cdf, on a $h_t - (-9.25)$ donc quand la vol vaut -9.25, on a cdf(0) = 0.5 donc $\mu(-9.25) = 10+1.5\times 0.5 = -9.25$
- Au dénominateur de la cdf, on a $ 0.5 \sigma / \sqrt{(1-\phi ^2)}$
  - Pour un processus AR(1) stationnaire : $h_{t+1} = c + \phi h_t + \sigma Z_t$, la variance est $Var(h_t) = \dfrac{\sigma ^2}{1-\phi^2}$ donc $\sigma_h = \dfrac{\sigma}{\sqrt{1-\phi^2}}$
  - On multiplie par 0.5 pour que la variable d'entrée de la cdf soit adimensionnelle

## Installation :

```
pip install -e .
```

## Arborescence

```
sv-spline/
├── pyproject.toml
├── src/sv_spline/
│ ├── spline.py # nœuds, Q, R, K_ncs, évaluation, pentes aux bords
│ ├── transitions.py # construction des m_h : droite, spline, vraie m_h du DGP
│ ├── filter.py # filtre bootstrap, méthodes de rééchantillonnage
│ ├── models.py # SV basique, SV semi-paramétrique
│ ├── proposals.py # marche aléatoire, proposition de g
│ ├── pmmh.py # l'échantillonneur générique
│ ├── simulation.py # DGP
│ ├── data.py # chargement du Bitcoin, calcul des rendements
│ └── diagnostics.py # taux, traces, IMSE ...
├── tests/
├── experiments/ # un script par expérience
├── notebooks/ # exploration et figures uniquement
├── data/ # données Bitcoin etc.
└── results/
```
