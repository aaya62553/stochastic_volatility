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
