#!/usr/bin/env python3
"""Generate the manuscript's mode illustrations and five closed-form comparisons.

Run beside verify_closed_forms.py and verify_quadrature.py with NumPy.
Outputs are deterministic CSV data and a JSON record under figure_results.
The parameter sweeps and conditional mode-conversion illustration are included
in software version 1.3.0. Conversion events and amplitudes are not predicted.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import json
import hashlib
import platform
import numpy as np
from verify_closed_forms import (
    FactorizedTiltedUniaxial as Reciprocal,
    QuadraticTiltedUniaxial as Quadratic,
    IndefiniteTiltedUniaxial as Hyperbolic,
    HeliconicalUniaxial as Heliconical,
    CoordinateShear,
    maxwell_residuals,
)
from verify_quadrature import exact_velocity, gauss_position, integrate_geodesic_rkf45

class RectangularShear(CoordinateShear):
    """Interior 0 <= z <= L of the manuscript's cubic shear profile."""
    length = 8.0
    delta_x = 0.6
    delta_y = 0.2
    @classmethod
    def f(cls, z):
        xi = z / cls.length
        return -cls.delta_x * (3 * xi**2 - 2 * xi**3)
    @classmethod
    def g(cls, z):
        xi = z / cls.length
        return -cls.delta_y * (3 * xi**2 - 2 * xi**3)
    @classmethod
    def f_prime(cls, z):
        xi = z / cls.length
        return -cls.delta_x / cls.length * (6 * xi - 6 * xi**2)
    @classmethod
    def g_prime(cls, z):
        xi = z / cls.length
        return -cls.delta_y / cls.length * (6 * xi - 6 * xi**2)

def variant(base, **attributes):
    return type(base.__name__ + "Variant", (base,), attributes)

def positions(cls, z, p):
    args = (z, np.zeros(2), 0.0)
    return cls.analytic_position(*args) if issubclass(cls, Heliconical) else cls.analytic_position(*args, p)

def write_csv(path, columns, values):
    np.savetxt(path, values, delimiter=",", header=",".join(columns), comments="", fmt="%.17g")

def heliconical_conversion_illustration(out, z_values):
    """Draw possible extraordinary paths after conversion at selected positions.

    The starting positions are illustrative, not predictions of mode coupling.
    Every path preserves transverse momentum p=0 at its starting point and
    subsequently remains extraordinary. No amplitude or intensity is assigned.
    """
    momentum = np.zeros(2)
    origin = np.zeros(2)
    start_indices = np.arange(40, len(z_values) - 1, 40)
    if len(z_values) != 241 or len(start_indices) != 5:
        raise AssertionError("The illustration requires six equal pitch intervals")
    columns = ["z"]
    values = [z_values]
    records = []
    delta_z = z_values - z_values[0]
    projection_columns = ["delta_z"]
    projection_values = [delta_z]
    coordinate_bound = 2.0 * abs(Heliconical.walkoff_coefficient() / Heliconical.twist_rate)
    if coordinate_bound >= 0.7:
        raise AssertionError("A full transverse projection can exceed the plot window")
    for index, start_index in enumerate(start_indices, 1):
        z0 = float(z_values[start_index])
        sample_z = z_values[start_index:]
        analytical = Heliconical.analytic_position(sample_z, origin, z0)
        quadrature = gauss_position(Heliconical.metric, sample_z, origin, z0,
                                   momentum, order=128)
        initial_slope = exact_velocity(Heliconical.metric, z0, momentum)
        grid, states, _ = integrate_geodesic_rkf45(
            Heliconical.metric, z0, float(z_values[-1]),
            np.r_[origin, initial_slope], sample_z, rtol=1e-12, atol=1e-14)
        if not np.array_equal(grid, sample_z):
            raise AssertionError("Unexpected conversion-illustration output grid")
        initial_error = float(np.max(np.abs(analytical[0] - origin)))
        rkf45_error = float(np.max(np.abs(analytical - states[:, :2])))
        gauss_error = float(np.max(np.abs(analytical - quadrature)))
        g0 = Heliconical.metric(z0)
        tangent = np.r_[initial_slope, 1.0]
        optical_factor = np.sqrt(tangent @ g0 @ tangent)
        reconstructed_p = (g0[:2, :2] @ initial_slope + g0[:2, 2]) / optical_factor
        momentum_error = float(np.max(np.abs(reconstructed_p - momentum)))
        if initial_error > 1e-14 or momentum_error > 1e-14:
            raise AssertionError((z0, "Incorrect initial condition", initial_error, momentum_error))
        if rkf45_error > 1e-10 or gauss_error > 1e-12:
            raise AssertionError((z0, "Incorrect extraordinary trajectory", rkf45_error, gauss_error))
        # Panel (d) continues each ray for a complete pitch from its own start.
        full_z = z0 + delta_z
        full_analytical = Heliconical.analytic_position(full_z, origin, z0)
        full_quadrature = gauss_position(
            Heliconical.metric, full_z, origin, z0, momentum, order=128)
        full_grid, full_states, _ = integrate_geodesic_rkf45(
            Heliconical.metric, z0, float(full_z[-1]),
            np.r_[origin, initial_slope], full_z, rtol=1e-12, atol=1e-14)
        if not np.array_equal(full_grid, full_z):
            raise AssertionError("Unexpected full-projection output grid")
        full_rkf45_error = float(np.max(np.abs(full_analytical - full_states[:, :2])))
        full_gauss_error = float(np.max(np.abs(full_analytical - full_quadrature)))
        closure_error = float(np.max(np.abs(full_analytical[-1] - full_analytical[0])))
        if full_rkf45_error > 1e-10 or full_gauss_error > 1e-12 or closure_error > 1e-14:
            raise AssertionError((z0, "Incorrect full projection",
                                  full_rkf45_error, full_gauss_error, closure_error))
        if np.max(np.abs(full_analytical)) > coordinate_bound + 1e-14:
            raise AssertionError("Unexpected projection coordinate bound")
        projection_columns.extend([f"z_e{index}", f"x_e{index}", f"y_e{index}"])
        projection_values.extend([full_z, full_analytical[:, 0], full_analytical[:, 1]])
        # NaN marks positions before this trajectory begins, not invalid ray data.
        padded = np.full((len(z_values), 2), np.nan)
        padded[start_index:] = analytical
        columns.extend([f"x_e{index}", f"y_e{index}"])
        values.extend([padded[:, 0], padded[:, 1]])
        records.append({
            "index": index, "start_z": z0, "start_position": [0.0, 0.0],
            "transverse_momentum": momentum.tolist(),
            "samples": len(sample_z), "initial_position_error": initial_error,
            "initial_momentum_error": momentum_error,
            "max_rkf45_error": rkf45_error, "max_gauss_error": gauss_error,
            "full_projection": {
                "z_interval": [z0, float(full_z[-1])],
                "samples": len(full_z),
                "max_rkf45_error": full_rkf45_error,
                "max_gauss_error": full_gauss_error,
                "closure_error": closure_error,
                "minimum_xy": full_analytical.min(axis=0).tolist(),
                "maximum_xy": full_analytical.max(axis=0).tolist(),
            },
        })
    write_csv(out / "demo_heliconical_conversions.csv", columns, np.column_stack(values))
    write_csv(out / "demo_heliconical_conversion_starts.csv", ["z", "x_o", "y_o"],
              np.column_stack([z_values[start_indices], np.zeros((5, 2))]))
    write_csv(out / "demo_heliconical_full_projections.csv", projection_columns,
              np.column_stack(projection_values))
    return {
        "interpretation": "Extraordinary trajectories conditional on one ordinary-to-extraordinary conversion per path at a selected point on the ordinary ray",
        "conversion_positions": "five equally spaced interior positions over one pitch",
        "spacing": float(z_values[-1] / 6.0),
        "conversion_amplitudes_calculated": False,
        "conversion_events_predicted": False,
        "subsequent_conversions_calculated": False,
        "projection_span": "one axial pitch from each selected starting position",
        "projection_axis_limits": [-0.7, 0.7],
        "full_projection_absolute_coordinate_bound": coordinate_bound,
        "archive_status": "Included in software version 1.3.0",
        "paths": records,
    }

def main():
    out = Path(__file__).resolve().parent / "figure_results"
    out.mkdir(exist_ok=True)
    z = np.linspace(0.0, 8.0, 161)
    q0 = np.zeros(2)
    cases = [
        ("reciprocal", Reciprocal, "gradient", [-0.06, 0.04, 0.18], [0.54, -0.29]),
        ("quadratic", Quadratic, "profile_rate", [0.04, 0.3, 1.2], [0.54, -0.29]),
        ("hyperbolic", Hyperbolic, "gradient", [-0.08, 0.04, 0.14], [1.8, -1.1]),
        ("heliconical", Heliconical, "cone_angle", [0.1, 0.5, 1.0], [0.0, 0.0]),
        ("rectangular", RectangularShear, "delta_x", [0.3, 0.6, 0.9], [0.0, 0.0]),
    ]
    report = {
        "provenance": "Manuscript parameter sweeps and paired-mode illustrations",
        "software_version": (Path(__file__).resolve().parent / "VERSION").read_text(encoding="utf-8").strip(),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ("generate_manuscript_figures.py", "verify_closed_forms.py", "verify_quadrature.py")
        },
        "protocol": {"z_interval": [0,8], "samples":161, "rtol":1e-12, "atol":1e-14,
                     "rkf45_safety_factor":0.9,"gauss_legendre_order":128,
                     "error": "maximum absolute transverse component difference over samples",
                     "rkf45_output": "Integration steps are shortened to reach each output point.",
                     "initial_position":[0,0]},
        "base_parameters":{
            "reciprocal_quadratic":{"epsilon_o_ref":1.9,"epsilon_e_ref":2.7,"axis":Reciprocal.axis.tolist(),"chi0":.78,"eta_c":.9,"z_c":2.3},
            "hyperbolic":{"epsilon_o_ref":-1.25,"epsilon_e_ref":2.6,"axis":Hyperbolic.axis.tolist(),"eta0":.8},
            "heliconical":{"epsilon_o":1.85,"epsilon_e":2.55,"Omega":.43,"phi0":.27},
            "rectangular":{"L":8,"delta_y":.2},
            "mu":1,
        },
        "cases":[],
    }
    maxima = []
    for index,(name, base, parameter, values, momentum) in enumerate(cases,1):
        p = np.array(momentum)
        errors = []
        for k,value in enumerate(values,1):
            cls = variant(base, **{parameter:value})
            exact = positions(cls,z,p)
            quadrature = gauss_position(cls.metric,z,q0,0,p,order=128)
            slope0 = exact_velocity(cls.metric,0,p)
            grid, states, stats = integrate_geodesic_rkf45(
                cls.metric,0,8,np.r_[q0,slope0],z,rtol=1e-12,atol=1e-14)
            if not np.array_equal(grid,z):
                raise AssertionError("Unexpected output grid")
            er = np.max(np.abs(exact-states[:,:2]),axis=1)
            eg = np.max(np.abs(exact-quadrature),axis=1)
            # Endpoint comparisons at a sparse output schedule avoid a
            # convergence conclusion caused only by densely prescribed outputs.
            sparse = []
            for tolerance in (1e-6,1e-9,1e-12):
                _, st, counts = integrate_geodesic_rkf45(
                    cls.metric,0,8,np.r_[q0,slope0],np.array([0.,8.]),
                    rtol=tolerance,atol=.01*tolerance)
                sparse.append({"rtol":tolerance,"endpoint_error":float(np.max(np.abs(st[-1,:2]-exact[-1]))),**counts})
            if name != "rectangular":
                sparse_errors = [run["endpoint_error"] for run in sparse]
                if not all(sparse_errors[i + 1] < sparse_errors[i] for i in range(2)):
                    raise AssertionError((name, value, "No sparse-output convergence", sparse_errors))
            if sparse[-1]["endpoint_error"] > 1e-10:
                raise AssertionError((name, value, "Sparse endpoint mismatch", sparse))
            margins = []
            for zi in z:
                g = cls.metric(zi); ai = np.linalg.inv(g[:2,:2])
                h = g[2,2]-g[2,:2]@ai@g[:2,2]
                d = 1-p@ai@p
                margins.append(h*d)
            if min(margins)<=0 or np.max(er)>1e-10 or np.max(eg)>1e-10:
                raise AssertionError((name,value,min(margins),max(er),max(eg)))
            write_csv(out/f"{name}_{k}.csv",
                      ["z","x","y","x_rkf45","y_rkf45","x_gauss","y_gauss","error_rkf45","error_gauss"],
                      np.column_stack([z,exact,states[:,:2],quadrature,er,eg]))
            item={"medium":name,"parameter":parameter,"value":value,"momentum":momentum,
                  "max_rkf45_error":float(max(er)),"max_gauss_error":float(max(eg)),
                  "minimum_sampled_hD":float(min(margins)),"sparse_endpoint_runs":sparse,**stats}
            if name != "rectangular":
                physical = maxwell_residuals(cls.metric, cls.dielectric, z, p)
                if (physical["maxwell_normalized_singularity_max_residual"] > 1e-12
                    or physical["quadratic_dispersion_max_abs_residual"] > 1e-11
                    or physical["poynting_direction_max_sine_error"] > 1e-10
                    or physical["minimum_normalized_axial_poynting_flux"] <= 0):
                    raise AssertionError((name, value, physical))
                item["maxwell"] = physical
            report["cases"].append(item)
            errors.append((max(er),max(eg)))
            print(name,parameter,value,"RKF45",max(er),"GL",max(eg),flush=True)
        maxima.append([index,*np.max(errors,axis=0)])
    write_csv(out/"max_errors.csv",["case","rkf45","gauss"],np.array(maxima))
    # Mode illustrations share the starting point and transverse wave covector,
    # not necessarily their initial energy-flow direction.
    demos = []
    for name,base in (("reciprocal",Reciprocal),("quadratic",Quadratic)):
        p=np.array([.54,-.29])
        ordinary=variant(base,base_metric=classmethod(lambda cls: cls.epsilon_o_0*np.eye(3)))
        extraordinary=positions(base,z,p)
        o=positions(ordinary,z,p)
        # Verify the ordinary specialization before presenting the paired rays.
        _,st,_=integrate_geodesic_rkf45(ordinary.metric,0,8,np.r_[q0,exact_velocity(ordinary.metric,0,p)],z,rtol=1e-12,atol=1e-14)
        err=float(np.max(np.abs(st[:,:2]-o)))
        if err>1e-10: raise AssertionError((name,err))
        write_csv(out/f"demo_{name}.csv",["z","x_o","y_o","x_e","y_e"],np.column_stack([z,o,extraordinary]))
        demos.append({"medium":name,"momentum":p.tolist(),"ordinary_rkf45_error":err})
    zh=np.linspace(0,2*np.pi/Heliconical.twist_rate,241)
    he=positions(Heliconical,zh,np.zeros(2))
    write_csv(out/"demo_heliconical.csv",["z","x_o","y_o","x_e","y_e"],np.column_stack([zh,np.zeros((len(zh),2)),he]))
    report["mode_demonstrations"]=demos+[{"medium":"heliconical","momentum":[0,0],"z_end":float(zh[-1]),"ordinary":"straight axial ray",
        "conditional_conversions":heliconical_conversion_illustration(out,zh)}]
    report["status"] = "passed"
    temporary = out / "verification.json.tmp"
    temporary.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    temporary.replace(out / "verification.json")

if __name__ == "__main__":
    main()
