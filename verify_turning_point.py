#!/usr/bin/env python3
"""Verify continuation through a z-turning point with Hamilton equations.

The z-parameterized quadrature is singular at the turning point.  The full
Hamiltonian system remains regular and is compared with an analytic parabola.
"""

from __future__ import annotations

import json

import numpy as np

from verify_quadrature import integrate_rkf45


def hamiltonian_rhs(state: np.ndarray, gradient: float, base_index_squared: float) -> np.ndarray:
    """Hamilton equations for G(z) = N(z) I and H = |P|^2/(2N)."""
    _, z, px, pz = state
    index_squared = base_index_squared - gradient * z
    momentum_squared = px * px + pz * pz
    return np.array(
        [
            px / index_squared,
            pz / index_squared,
            0.0,
            -gradient * momentum_squared / (2.0 * index_squared * index_squared),
        ]
    )


def integrate_until_return(tolerance: float) -> tuple[np.ndarray, dict[str, int]]:
    base_index_squared = 2.0
    gradient = 0.5
    px = 1.0
    pz = np.sqrt(base_index_squared - px * px)
    state = np.array([0.0, 0.0, px, pz])
    # The exact return x=8 occurs at Hamiltonian parameter tau=32/3.
    tau = np.linspace(0.0, 32.0 / 3.0, 2001)
    _, states, statistics = integrate_rkf45(
        lambda _tau, current: hamiltonian_rhs(current, gradient, base_index_squared),
        float(tau[0]),
        float(tau[-1]),
        state,
        tau,
        rtol=tolerance,
        atol=0.01 * tolerance,
    )
    return states, statistics


def diagnostics(states: np.ndarray) -> dict[str, float]:
    base_index_squared = 2.0
    gradient = 0.5
    px = 1.0
    turning_z = (base_index_squared - px * px) / gradient
    turning_x = 2.0 * px * np.sqrt(base_index_squared - px * px) / gradient

    x = states[:, 0]
    z = states[:, 1]
    pz = states[:, 3]
    exact_z = turning_z - gradient * (x - turning_x) ** 2 / (4.0 * px * px)
    path_error = np.max(np.abs(z - exact_z))

    index_squared = base_index_squared - gradient * z
    hamiltonian = (states[:, 2] ** 2 + pz**2) / (2.0 * index_squared)
    hamiltonian_drift = np.max(np.abs(hamiltonian - 0.5))

    turning_index = int(np.argmin(np.abs(pz)))
    return {
        "parabolic_path_max_abs_error": float(path_error),
        "hamiltonian_max_abs_drift": float(hamiltonian_drift),
        "turning_x_abs_error": float(abs(x[turning_index] - turning_x)),
        "turning_z_abs_error": float(abs(z[turning_index] - turning_z)),
        "minimum_abs_vertical_momentum": float(abs(pz[turning_index])),
        "minimum_vertical_momentum": float(np.min(pz)),
        "maximum_vertical_momentum": float(np.max(pz)),
    }


def run_checks() -> dict[str, object]:
    results: dict[str, object] = {}
    for tolerance in (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12):
        states, statistics = integrate_until_return(tolerance)
        results[f"{tolerance:.0e}"] = {**diagnostics(states), **statistics}

    finest = results["1e-12"]
    assert finest["parabolic_path_max_abs_error"] < 2.0e-12
    assert finest["hamiltonian_max_abs_drift"] < 2.0e-12
    assert finest["minimum_vertical_momentum"] < 0.0 < finest["maximum_vertical_momentum"]
    return results


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
