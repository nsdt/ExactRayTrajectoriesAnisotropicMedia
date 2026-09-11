#!/usr/bin/env python3
"""Generate the manuscript's mode illustrations and five closed-form comparisons.

Run beside verify_closed_forms.py and verify_quadrature.py with NumPy.
Outputs are deterministic CSV data and a JSON record under figure_results.
The parameter sweeps are included in software version 1.2.0 and later.
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
    report["mode_demonstrations"]=demos+[{"medium":"heliconical","momentum":[0,0],"z_end":float(zh[-1]),"ordinary":"straight axial ray"}]
    report["status"] = "passed"
    temporary = out / "verification.json.tmp"
    temporary.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    temporary.replace(out / "verification.json")

if __name__ == "__main__":
    main()
