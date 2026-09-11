# Exact ray trajectories: numerical experiments

[![DOI (all archived versions)](https://zenodo.org/badge/DOI/10.5281/zenodo.22037537.svg)](https://doi.org/10.5281/zenodo.22037537)

Code and data for *Exact ray trajectories in anisotropic media with axially
varying non-diagonal optical metrics*.

Version **1.2.0** is prepared for release. Its version-specific Zenodo DOI
will be added after archival; the badge links to the existing version series.

## Reproduce

Requires Python 3.12 and NumPy 1.26.4. The recorded results were obtained with
CPython 3.12.3 on Ubuntu/WSL.

```sh
git clone https://github.com/nsdt/ExactRayTrajectoriesAnisotropicMedia.git
cd ExactRayTrajectoriesAnisotropicMedia
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_numerical_experiments.py
```

On Windows PowerShell, use `python` to create the environment and
`.venv\Scripts\python.exe` in place of `.venv/bin/python`.

The final command runs the baseline checks and all manuscript-figure checks.
It regenerates 19 CSV files in `figure_results/`,
`results/numerical-experiments.json`, and `figure_results/verification.json`.
It exits with an error if any check fails. All parameters and tolerances are
fixed in the scripts. Last digits can differ between numerical-library or
platform builds; the checks use numerical acceptance thresholds.

## Files

| File | Purpose |
|---|---|
| `run_numerical_experiments.py` | Run all checks and regenerate outputs |
| `verify_quadrature.py` | General non-diagonal metric, tolerance study, endpoint inversion, diagonal limits |
| `verify_turning_point.py` | Hamiltonian continuation through a turning point and a separate tolerance study |
| `verify_uniaxial_branch.py` | Ordinary and extraordinary dispersion and Maxwell checks |
| `verify_closed_forms.py` | Five closed-form solutions, specialized inverses, stable zero-parameter limits, and Poynting directions |
| `generate_manuscript_figures.py` | Regenerate and verify the two manuscript figures |
| `figure_results/*.csv`, `figure_results/verification.json` | Fifteen comparisons, paired-mode trajectories, errors, parameters, and source hashes |
| `mode_demonstrations.tex`, `closed_form_verification.tex` | PGFPlots sources for the two manuscript figures |
| `mode_demonstrations.pdf`, `closed_form_verification.pdf` | Rendered figures |
| `results/numerical-experiments.json` | Recorded errors, residuals, and runtime information |
| `CITATION.cff`, `VERSION`, `LICENSE` | Citation metadata, version, and MIT license |

## Manuscript figures

The revised manuscript separates illustrations of ordinary and extraordinary
rays from verification of all five closed-form solutions, with three parameter values
per solution. The main command above includes these calculations. To run them
separately, use `.venv/bin/python generate_manuscript_figures.py`.
See [MANUSCRIPT_FIGURES.md](MANUSCRIPT_FIGURES.md) for the parameters and figure
build commands. PDFs are included for inspection; rebuilding them requires
LaTeX with PGFPlots, whereas the numerical checks require only Python and NumPy.

## Changes for 1.2.0

- Five closed-form solutions compared with independent geodesic integration
  and Gauss-Legendre quadrature at three parameter values each.
- Separate illustrations of ordinary and extraordinary rays.
- Nineteen new CSVs, a complete figure-verification report, two PGFPlots
  sources, and their rendered PDFs.
- A single entry point for the baseline and manuscript-figure calculations.
- Removed the superseded figure generator and four CSV files used only by the
  previous combined figure. They remain available in the earlier releases.

## Changes for 1.1.0

- Stable reciprocal-profile evaluation at zero and small transverse momentum
  and grading rate.
- Quadratic-scale closed-form and inverse-endpoint checks.
- Independent axial-covector, quadratic-dispersion, and Poynting-direction checks.
- Turning-point convergence checks without a dense output grid fixing the step size.

## Citation

The version-specific DOI for **v1.2.0** is pending. Until it is assigned,
identify this version and the exact Git commit when referring to these files.

Previous archives remain available as
[v1.1.0](https://doi.org/10.5281/zenodo.22654497) and
[v1.0.0](https://doi.org/10.5281/zenodo.22037538).
Neither contains the new manuscript-figure calculations in version 1.2.0.
