# Exact ray trajectories: numerical experiments

This is a self-contained release of the numerical experiments supporting
*Exact ray trajectories in anisotropic media with axially varying non-diagonal
optical metrics*. It contains no manuscript source or output. The only runtime
dependency is NumPy; the adaptive RKF45 integrator and Gauss--Legendre
quadrature are implemented in the included Python files.

## Validated environment

The committed results were generated and checked with:

- CPython 3.12.3
- NumPy 1.26.4 (pinned exactly in `requirements.txt`)

Other Python versions may work, but the acceptance thresholds below were
validated with this exact combination.

## Contents and experiment purposes

- `verify_quadrature.py`: compares the conserved-momentum Gauss--Legendre
  quadrature with an independent adaptive three-dimensional geodesic
  integration; checks endpoint inversion, momentum conservation, a rotating
  tilt closed form, and two diagonal reductions.
- `verify_turning_point.py`: continues a ray through a vertical turning point
  using the regular Hamilton system and compares it with the analytic
  parabola while monitoring Hamiltonian drift.
- `verify_uniaxial_branch.py`: checks ordinary and extraordinary Maxwell
  branches against the metric block velocity, and compares the extraordinary
  quadrature with geodesic integration.
- `verify_closed_forms.py`: checks heliconical, factorized tilted-uniaxial,
  epsilon-negative indefinite, and coordinate-shear closed forms against the
  general quadrature and independent geodesic integration.
- `generate_figure_data.py`: deterministically regenerates the four CSV files
  in `data/`.
- `run_numerical_experiments.py`: runs all four verification modules and writes
  `results/numerical-experiments.json` only after every assertion passes.

## Fixed parameters

All parameters are hard-coded in the scripts so a run needs no external input.

- General quadrature: `z=[0,8]`, initial `q=(0.10,-0.20)`, initial slope
  `(0.12,-0.08)`, endpoint target `(0.9,-0.6)`, 64-point Gauss--Legendre
  quadrature, and RKF45 relative tolerances `1e-6`, `1e-8`, `1e-10`, `1e-12`
  with absolute tolerance equal to 1% of the relative tolerance. The test
  tensor has eigenvalues `1.40+0.20 sin(0.70z)`,
  `0.90+0.10 cos(0.40z)`, and `1.80+0.15 sin(0.50z+0.20)`, rotated by
  `Rz(0.25z) Ry(0.18 sin(0.60z)) Rx(0.12z)`. The rotating-tilt constants are
  `(a,d,c,phi,kappa)=(1.25,0.85,2.10,0.63,0.37)`. Diagonal reductions use
  `z=[0.20,4.70]` at 101 samples. Case A is
  `diag(1.30,0.75,1.60+0.25 sin(0.70z))` with slope `(0.18,-0.11)`; case B is
  `diag(g,-g,1.70+0.20 sin(0.55z))`, where
  `g=1.10+0.16 cos(0.45z)`, with slope `(0.14,-0.14)`.
- Turning point: isotropic `N(z)=2.0-0.5z`, initial momenta `px=pz=1`, Hamilton
  parameter `[0,32/3]` sampled at 2001 points, with the same four tolerance
  levels.
- Uniaxial branches: `z=[0,7]` at 161 samples; extraordinary initial slope
  `(0.13,-0.09)`, ordinary initial slope `(-0.08,0.11)`, and extraordinary
  initial position `(0.17,-0.24)`. The profiles are
  `epsilon_o=1.65+0.10 sin(0.40z)`,
  `epsilon_e=2.25+0.12 cos(0.30z+0.20)`, and
  `mu=1.05+0.03 sin(0.20z-0.10)`; the optical-axis angles are
  `theta=0.58+0.17 sin(0.35z)` and `phi=0.22z+0.08 sin(0.51z)`.
- Closed forms: `z=[0,8]` at 161 samples. Heliconical constants are
  `(epsilon_o,epsilon_e,cone_angle,twist_rate,initial_azimuth) =
  (1.85,2.55,0.61,0.43,0.27)`. The factorized case uses
  `(epsilon_o_0,epsilon_e_0,u0,gradient)=(1.90,2.70,0.78,0.041)` and axis
  proportional to `(0.48,-0.37,0.795)`. The indefinite case uses
  `(-1.25,2.60,0.80,0.025)` and axis proportional to
  `(0.26,-0.19,0.947)`. Initial `(q,p)` pairs are heliconical
  `((0.13,-0.19),(0,0))`, factorized
  `((-0.21,0.14),(0.54,-0.29))`, indefinite
  `((-0.08,0.12),(1.80,-1.10))`, and coordinate shear
  `((0.07,-0.11),(0.34,-0.23))`. The shear functions are
  `f(z)=0.31 sin(0.52z)+0.018z^2` and
  `g(z)=0.24(1-cos(0.41z))-0.027z`.
- Figure CSVs: factorized momenta `(0.54,-0.29)` and `(0.31,0.24)`;
  indefinite momenta `(1.80,-1.10)` and `(1.60,0.90)`; heliconical
  `(cone angle, twist rate)` pairs `(0.48,0.35)`, `(0.61,0.43)`, and
  `(0.72,0.55)`; rectangular-shear amplitudes `0.30`, `0.60`, and `0.90`.

## Run in a clean virtual environment

From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_numerical_experiments.py
```

The last command exits nonzero on any failed assertion. On success it replaces
`results/numerical-experiments.json` atomically and prints its path. Equivalent
Make targets are `make setup`, `make experiments`, and `make verify`.

Regenerate figure data with:

```sh
.venv/bin/python generate_figure_data.py
```

This overwrites only the four tracked CSV files in `data/`.

## Numerical acceptance criteria

A complete run passes only when every criterion encoded as an assertion holds:

- Quadrature: finest endpoint, sampled position, sampled slope, and momentum
  errors are each `<2e-10`; endpoint Newton inversion takes exactly 5 map
  evaluations, has residual `<1e-12`, and sampled minimum `D>0.88`; rotating
  tilt error is `<2e-12`; both diagonal-reduction errors are `<2e-13`.
- Turning point: finest parabolic-path error and Hamiltonian drift are each
  `<2e-12`, and vertical momentum takes both positive and negative values.
- Uniaxial branches: block/Hamilton velocity and normalized Maxwell
  singularity residuals are `<2e-13` for both modes; extraordinary
  quadrature/geodesic position and velocity/geodesic slope errors are
  `<3e-10`; the returned sample grid error is `<1e-14`.
- Closed forms: every analytic/quadrature error is `<2e-12` and every
  analytic/geodesic position error is `<4e-10`; normalized Maxwell residuals
  are `<2e-13` for heliconical and factorized cases and `<5e-13` for the
  indefinite case; coordinate-shear inverse-momentum error is `<2e-14`.

Raw diagnostic values, including quantities without a pass/fail threshold,
are preserved in `results/numerical-experiments.json`.

## Archive and citation

The numerical experiments used for the reported results are archived as
version `v1.0.0` on Zenodo:

[![DOI](https://zenodo.org/badge/1341342488.svg)](https://doi.org/10.5281/zenodo.22037537)

To reproduce or cite the version used for the article, use:

> Nishidate, Y. (2026). *Exact ray trajectories in anisotropic media with axially varying non-diagonal optical metrics:
> numerical experiments* (Version 1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.22037538

The source repository is
[github.com/nsdt/ExactRayTrajectoriesAnisotropicMedia](https://github.com/nsdt/ExactRayTrajectoriesAnisotropicMedia).

## License and citation

The code and data are released under the MIT License; see `LICENSE`. Citation
metadata without an unassigned DOI is provided in `CITATION.cff` and
`.zenodo.json`.
