# Exact ray trajectories: numerical experiments

Code and data for *Exact ray trajectories in anisotropic media with axially
varying non-diagonal optical metrics*.

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

The final command runs all checks, regenerates the four CSV files in `data/`,
and writes `results/numerical-experiments.json`. It exits with an error if any
check fails. All parameters and tolerances are fixed in the scripts.

## Files

| File | Purpose |
|---|---|
| `run_numerical_experiments.py` | Run all checks and regenerate outputs |
| `verify_quadrature.py` | General non-diagonal metric, tolerance study, endpoint inversion, diagonal limits |
| `verify_turning_point.py` | Hamiltonian continuation through a turning point and a separate tolerance study |
| `verify_uniaxial_branch.py` | Ordinary and extraordinary dispersion and Maxwell checks |
| `verify_closed_forms.py` | Five closed forms, specialized inverses, stable zero-parameter limits, and Poynting directions |
| `generate_figure_data.py` | Regenerate figure CSVs separately |
| `data/*.csv` | Coordinates for the representative-trajectory figure |
| `results/numerical-experiments.json` | Recorded errors, residuals, and runtime information |
| `CITATION.cff`, `VERSION`, `LICENSE` | Citation metadata, version, and MIT license |

## Changes for 1.1.0

- Stable reciprocal-profile evaluation at zero and small transverse momentum
  and grading rate.
- Quadratic-scale closed-form and inverse-endpoint checks.
- Independent axial-covector, quadratic-dispersion, and Poynting-direction checks.
- Turning-point convergence checks without a dense output grid fixing the step size.

## Citation

This revision is prepared for **v1.1.0**; its GitHub release and Zenodo DOI
are pending. Cite the version-specific DOI once that release is archived.

The preceding **v1.0.0** is archived at
[Zenodo](https://doi.org/10.5281/zenodo.22037538).
That archive does not include the changes listed above.
