#!/usr/bin/env python3
"""Verify the physical uniaxial realization of the block integral.

The extraordinary and ordinary wave-covector solutions are obtained directly
from the Maxwell plane-wave equation.  Their Hamiltonian ray velocities
are compared with the metric block formula.  For the extraordinary mode,
the resulting integral is also compared with an independent integration of
the geodesic equation in the corresponding ray metric.
"""

from __future__ import annotations

import json

import numpy as np

from verify_quadrature import (
    exact_velocity,
    gauss_position,
    initial_momentum,
    integrate_geodesic_rkf45,
)


Array = np.ndarray


def material_parameters(z: complex | float) -> tuple[complex | float, complex | float, complex | float]:
    """Smooth positive material parameters."""
    epsilon_o = 1.65 + 0.10 * np.sin(0.40 * z)
    epsilon_e = 2.25 + 0.12 * np.cos(0.30 * z + 0.20)
    permeability = 1.05 + 0.03 * np.sin(0.20 * z - 0.10)
    return epsilon_o, epsilon_e, permeability


def optic_axis(z: complex | float) -> Array:
    """A unit optical axis whose polar and azimuthal angles both vary."""
    theta = 0.58 + 0.17 * np.sin(0.35 * z)
    phi = 0.22 * z + 0.08 * np.sin(0.51 * z)
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def dielectric(z: complex | float) -> Array:
    epsilon_o, epsilon_e, _ = material_parameters(z)
    axis = optic_axis(z)
    projector = np.outer(axis, axis)
    return epsilon_o * (np.eye(3) - projector) + epsilon_e * projector


def extraordinary_metric(z: complex | float) -> Array:
    """Ray metric G_e for the extraordinary uniaxial mode."""
    epsilon_o, epsilon_e, permeability = material_parameters(z)
    axis = optic_axis(z)
    projector = np.outer(axis, axis)
    return permeability * (
        epsilon_e * (np.eye(3) - projector) + epsilon_o * projector
    )


def ordinary_metric(z: complex | float) -> Array:
    epsilon_o, _, permeability = material_parameters(z)
    return permeability * epsilon_o * np.eye(3)


def vertical_root(metric, z: float, transverse_momentum: Array, sign: float = 1.0) -> float:
    """Solve P^T G^{-1} P = 1 for the vertical covector."""
    inverse_metric = np.linalg.inv(np.asarray(metric(z), dtype=float))
    alpha = inverse_metric[2, 2]
    beta = inverse_metric[2, :2] @ transverse_momentum
    gamma = transverse_momentum @ inverse_metric[:2, :2] @ transverse_momentum - 1.0
    discriminant = beta * beta - alpha * gamma
    if discriminant <= 0.0:
        raise ValueError(f"no simple real vertical root at z={z}: discriminant={discriminant}")
    return float((-beta + sign * np.sqrt(discriminant)) / alpha)


def hamilton_velocity(metric, z: float, transverse_momentum: Array, sign: float = 1.0) -> Array:
    """dq/dz = Phi_p/Phi_r for Phi=(P^T G^{-1}P-1)/2."""
    inverse_metric = np.linalg.inv(np.asarray(metric(z), dtype=float))
    vertical_momentum = vertical_root(metric, z, transverse_momentum, sign)
    covector = np.r_[transverse_momentum, vertical_momentum]
    hamilton_velocity_3d = inverse_metric @ covector
    return hamilton_velocity_3d[:2] / hamilton_velocity_3d[2]


def maxwell_diagnostics(metric, z_values: Array, transverse_momentum: Array) -> dict[str, float]:
    """Evaluate a scale-independent singularity residual for the Maxwell matrix."""
    normalized_singularity_residuals = []
    dispersion_residuals = []
    for z in z_values:
        inverse_metric = np.linalg.inv(np.asarray(metric(z), dtype=float))
        vertical_momentum = vertical_root(metric, float(z), transverse_momentum)
        covector = np.r_[transverse_momentum, vertical_momentum]
        epsilon_o, _, permeability = material_parameters(z)
        maxwell_matrix = (
            (covector @ covector) * np.eye(3)
            - np.outer(covector, covector)
            - permeability * np.asarray(dielectric(z), dtype=float)
        )
        singular_values = np.linalg.svd(maxwell_matrix, compute_uv=False)
        normalized_singularity_residuals.append(
            singular_values[-1] / singular_values[0]
        )
        dispersion_residuals.append(abs(covector @ inverse_metric @ covector - 1.0))
        if metric is ordinary_metric:
            expected = permeability * epsilon_o
            dispersion_residuals.append(abs(covector @ covector - expected))
    return {
        "maxwell_normalized_singularity_max_residual": float(
            max(normalized_singularity_residuals)
        ),
        "quadratic_dispersion_max_abs_residual": float(max(dispersion_residuals)),
    }


def mode_checks(metric, z_values: Array, slope0: Array) -> tuple[Array, dict[str, float]]:
    transverse_momentum, _ = initial_momentum(metric, float(z_values[0]), slope0)
    block_velocities = np.array(
        [exact_velocity(metric, float(z), transverse_momentum) for z in z_values]
    )
    hamilton_velocities = np.array(
        [hamilton_velocity(metric, float(z), transverse_momentum) for z in z_values]
    )
    diagnostics = maxwell_diagnostics(metric, z_values, transverse_momentum)
    diagnostics["block_vs_hamilton_velocity_max_abs_error"] = float(
        np.max(np.abs(block_velocities - hamilton_velocities))
    )
    return transverse_momentum, diagnostics


def run_checks() -> dict[str, object]:
    z0, z1 = 0.0, 7.0
    sample_z = np.linspace(z0, z1, 161)
    extraordinary_slope0 = np.array([0.13, -0.09])
    ordinary_slope0 = np.array([-0.08, 0.11])

    extraordinary_p, extraordinary = mode_checks(
        extraordinary_metric, sample_z, extraordinary_slope0
    )
    _, ordinary = mode_checks(ordinary_metric, sample_z, ordinary_slope0)

    q0 = np.array([0.17, -0.24])
    initial_state = np.r_[q0, extraordinary_slope0]
    reference_positions = gauss_position(
        extraordinary_metric, sample_z, q0, z0, extraordinary_p
    )
    geodesic_z, geodesic_states, _ = integrate_geodesic_rkf45(
        extraordinary_metric,
        z0,
        z1,
        initial_state,
        sample_z,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    extraordinary["hamilton_quadrature_vs_geodesic_position_max_abs_error"] = float(
        np.max(np.abs(reference_positions - geodesic_states[:, :2]))
    )
    reference_slopes = np.array(
        [hamilton_velocity(extraordinary_metric, float(z), extraordinary_p) for z in sample_z]
    )
    extraordinary["hamilton_velocity_vs_geodesic_slope_max_abs_error"] = float(
        np.max(np.abs(reference_slopes - geodesic_states[:, 2:]))
    )
    extraordinary["minimum_metric_eigenvalue"] = float(
        min(np.linalg.eigvalsh(extraordinary_metric(z)).min() for z in sample_z)
    )
    extraordinary["maximum_absolute_g13_or_g23"] = float(
        max(np.max(np.abs(extraordinary_metric(z)[:2, 2])) for z in sample_z)
    )

    assert extraordinary["block_vs_hamilton_velocity_max_abs_error"] < 2.0e-13
    assert extraordinary["maxwell_normalized_singularity_max_residual"] < 2.0e-13
    assert extraordinary["hamilton_quadrature_vs_geodesic_position_max_abs_error"] < 3.0e-10
    assert extraordinary["hamilton_velocity_vs_geodesic_slope_max_abs_error"] < 3.0e-10
    assert ordinary["block_vs_hamilton_velocity_max_abs_error"] < 2.0e-13
    assert ordinary["maxwell_normalized_singularity_max_residual"] < 2.0e-13

    assert np.max(np.abs(geodesic_z - sample_z)) < 1.0e-14
    return {
        "extraordinary_mode_with_rotating_axis": extraordinary,
        "ordinary_mode": ordinary,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
