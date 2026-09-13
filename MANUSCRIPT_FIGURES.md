# Manuscript figures

Version 1.3.0 includes the parameter sweeps, paired-mode figures, and additional
trajectories after assumed mode conversion, including complete transverse
projections. Version 1.2.0 (DOI 10.5281/zenodo.22703136) contains the original
parameter sweeps and figures but not the additional trajectories or revised figures.
The main command, `python run_numerical_experiments.py`, includes all the
calculations below. They can also be run separately.

Run from an environment with the pinned NumPy dependency:

```sh
python generate_manuscript_figures.py
```

The command writes figure_results/verification.json and 22 CSV files:
15 trajectory comparisons, three paired-mode illustrations, one error summary,
and three files for the assumed-conversion trajectories, their starting points,
and complete transverse projections.
It exits with an error if a trajectory, initial-condition check, ordinary-mode
specialization, sparse-output convergence check, or uniaxial Maxwell check fails.

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

In Fig. 1(c) and (d), orange dashed curves illustrate extraordinary trajectories
after an assumed conversion from the ordinary mode at z=j*(2*pi/Omega)/6,
j=1,...,5. Each trajectory starts on the axial ordinary ray with transverse
momentum p=0 and follows the heliconical closed-form solution from that position.
The existing axis phase phi(z)=Omega*z+phi_ref is retained for every starting point.
Each added path remains extraordinary thereafter. The selected positions are
illustration parameters; no mode-conversion locations or amplitudes are predicted.

Each of the five added trajectories is compared with independent RKF45 geodesic
integration and Gauss-Legendre quadrature, and its starting position and momentum
are checked. The JSON records these checks under
mode_demonstrations[-1].conditional_conversions. These comparisons validate the
trajectories after the assumed events, not the occurrence of mode conversion.
NaN entries in demo_heliconical_conversions.csv occur only before the associated
trajectory starts. The 241-point grid covers one full pitch, with starts at
indices 40, 80, 120, 160, and 200. Orange markers are read from
demo_heliconical_conversion_starts.csv.

Fig. 1(c) retains the common interval 0 <= z <= P, where P=2*pi/Omega.
For Fig. 1(d), demo_heliconical_full_projections.csv extends each extraordinary
trajectory from its own starting position z_j to z_j+P (241 samples per path).
The columns record delta_z and the absolute z, x, and y of each path; the phase
is evaluated at the absolute z. Both axes span [-0.7, 0.7] with equal scales.
For these parameters, 2*abs(w/Omega)=0.658873 bounds each absolute transverse
coordinate, so the complete circular projections fit within the window.
The extended paths are independently compared with RKF45 and Gauss-Legendre
quadrature; their closure errors and coordinate extrema are recorded in each
path's full_projection entry in verification.json.

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
  reciprocal, quadratic, and heliconical media, with additional conditional
  extraordinary trajectories and their complete transverse projections.
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
The version-specific DOI for 1.3.0 will be recorded in the repository metadata
after archival. No DOI is assigned by these scripts.

The current Fig. 2 arranges its six panels in two columns and three rows.
The PGFPlots width and height settings are 7 cm and 4.8 cm;
the manuscript displays the complete figure at text width. The y-axis labels
are shifted 3 pt toward the axes. This layout revision changes neither the
numerical data nor the analytical/numerical comparisons.
