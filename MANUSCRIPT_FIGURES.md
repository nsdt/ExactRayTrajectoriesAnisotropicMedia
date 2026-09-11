# Manuscript figures for version 1.2.0

These experiments are included in version 1.2.0. The preceding v1.1.0 archive,
DOI 10.5281/zenodo.22654497, does not contain these parameter sweeps or figures.
The main command, `python run_numerical_experiments.py`, includes all the
calculations below. They can also be run separately.

Run from an environment with the pinned NumPy dependency:

```sh
python generate_manuscript_figures.py
```

The command writes figure_results/verification.json and 19 CSV files:
15 trajectory comparisons, three paired-mode illustrations, and one error
summary. It exits with an error if a trajectory, ordinary-mode specialization,
sparse-output convergence check, or uniaxial Maxwell check fails.

The five comparisons use reciprocal grading, quadratic grading, linear
hyperbolic grading, heliconical rotation, and the manuscript's cubic rectangular
beam shifter, each with three parameter values. Both transverse components are
checked at 161 equally spaced positions on 0 <= z <= 8. RKF45 uses rtol=1e-12
and atol=1e-14; Gauss-Legendre quadrature uses 128 nodes. Sparse runs with only
the entrance and exit requested use rtol=1e-6, 1e-9, and 1e-12 with
atol=0.01*rtol. The RKF45 step-size safety factor is 0.9, not 0.01.

The parameter values for the verification figure are:

| Medium | Varied parameter | Values |
|---|---|---|
| Reciprocal profile | gamma | -0.06, 0.04, 0.18 |
| Quadratic profile | lambda | 0.04, 0.3, 1.2 |
| Hyperbolic medium | gamma | -0.08, 0.04, 0.14 |
| Heliconical medium | theta | 0.1, 0.5, 1.0 radians |
| Rectangular beam shifter | Delta_x | 0.3, 0.6, 0.9 |

The mode illustration is a separate calculation. Its reciprocal, quadratic,
and heliconical parameters remain gamma=0.041, lambda=0.23, and theta=0.61.

The error is an **absolute difference**, not a relative difference: for each
parameter value, it is the maximum of abs(x_analytical - x_numerical) and
abs(y_analytical - y_numerical) over the 161 output positions. Panel (f)
further maximizes over the three parameter values for each medium.
The current overall maxima are 1.3543e-11 for RKF45 and 3.3751e-14 for
Gauss-Legendre quadrature. Relative errors are not reported because the
coordinates start at zero and can cross zero.

Sparse-output tolerance calculations remain auxiliary implementation checks.
The manuscript reports the trajectory comparisons at the 161 output positions.

The figures distinguish purpose:
- mode_demonstrations.tex: ordinary and extraordinary trajectories in the
  reciprocal, quadratic, and heliconical media; one pitch is shown for the helix.
- closed_form_verification.tex: all five closed-form solutions and independent numerical
  comparisons, plus maximum errors in both transverse coordinates.

Each illustrated ordinary/extraordinary pair shares the initial position and
transverse momentum, not necessarily its initial ray direction. The hyperbolic
example with epsilon_o<0 and mu>0 has no propagating real ordinary mode.
The transformation medium is nonbirefringent.

The JSON record includes every fixed/varied parameter, measured differences,
RKF45 step statistics, physical residuals, runtime versions, and hashes of the
three Python source files. The original baseline verification scripts and
results remain available separately.

To build the figure PDFs, a TeX installation with standalone, PGFPlots,
Latin Modern, and the usual AMS packages is required:

```sh
pdflatex -interaction=nonstopmode -halt-on-error mode_demonstrations.tex
pdflatex -interaction=nonstopmode -halt-on-error closed_form_verification.tex
```

Rebuilding the PDFs is separate from the Python calculation. The provided
PDFs were checked against the current PGFPlots sources and data.
Numerical comparisons use acceptance thresholds, not bitwise equality.
Floating-point differences in the last digits may occur across environments.
Version 1.2.0 is archived at https://doi.org/10.5281/zenodo.22703136.
The DOI was added to the repository metadata after archival; the release tag
and archived files are unchanged. No DOI is assigned by these scripts.
