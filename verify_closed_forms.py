#!/usr/bin/env python3
"""Verify physical closed-form trajectories of the block integral.

The checks cover reciprocal and quadratic grading, a heliconical
extraordinary ray, a real indefinite uniaxial trajectory, and a
coordinate-shear transformation medium. They also test the two endpoint
inverses, zero-parameter stability, and the uniaxial energy-flow direction.
Each analytic trajectory is compared with the general integral and with an
independent integration of the full three-dimensional geodesic equation.
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


def _sample_geodesic(metric, z_values: Array, q0: Array, slope0: Array) -> Array:
    z0 = float(z_values[0])
    z1 = float(z_values[-1])
    grid, states, _ = integrate_geodesic_rkf45(
        metric,
        z0,
        z1,
        np.r_[q0, slope0],
        z_values,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    if np.max(np.abs(grid - z_values)) > 2.0e-14:
        raise RuntimeError("RKF45 did not return the requested comparison points")
    return states


class HeliconicalUniaxial:
    """A uniaxial extraordinary metric with a heliconical optical axis."""

    epsilon_o = 1.85
    epsilon_e = 2.55
    cone_angle = 0.61
    twist_rate = 0.43
    initial_azimuth = 0.27

    @classmethod
    def axis(cls, z: complex | float) -> Array:
        phase = cls.twist_rate * z + cls.initial_azimuth
        sine = np.sin(cls.cone_angle)
        cosine = np.cos(cls.cone_angle)
        return np.array([sine * np.cos(phase), sine * np.sin(phase), cosine])

    @classmethod
    def metric(cls, z: complex | float) -> Array:
        axis = cls.axis(z)
        projector = np.outer(axis, axis)
        return cls.epsilon_e * (np.eye(3) - projector) + cls.epsilon_o * projector

    @classmethod
    def dielectric(cls, z: complex | float) -> Array:
        axis = cls.axis(z)
        projector = np.outer(axis, axis)
        return cls.epsilon_o * (np.eye(3) - projector) + cls.epsilon_e * projector

    @classmethod
    def walkoff_coefficient(cls) -> float:
        sine = np.sin(cls.cone_angle)
        cosine = np.cos(cls.cone_angle)
        delta = cls.epsilon_o - cls.epsilon_e
        transverse_entry = cls.epsilon_e + delta * sine * sine
        return float(delta * sine * cosine / transverse_entry)

    @classmethod
    def analytic_position(cls, z_values: Array, q0: Array, z0: float) -> Array:
        phase = cls.twist_rate * z_values + cls.initial_azimuth
        phase0 = cls.twist_rate * z0 + cls.initial_azimuth
        coefficient = cls.walkoff_coefficient() / cls.twist_rate
        displacement = coefficient * np.column_stack(
            [np.sin(phase0) - np.sin(phase), np.cos(phase) - np.cos(phase0)]
        )
        return q0 + displacement


class FactorizedTiltedUniaxial:
    """A fixed-axis uniaxial mode multiplied by a common axial scale."""

    epsilon_o_0 = 1.90
    epsilon_e_0 = 2.70
    u0 = 0.78
    gradient = 0.041
    axis = np.array([0.48, -0.37, 0.795])
    axis = axis / np.linalg.norm(axis)

    @classmethod
    def base_metric(cls) -> Array:
        projector = np.outer(cls.axis, cls.axis)
        return cls.epsilon_e_0 * (np.eye(3) - projector) + cls.epsilon_o_0 * projector

    @classmethod
    def base_dielectric(cls) -> Array:
        projector = np.outer(cls.axis, cls.axis)
        return cls.epsilon_o_0 * (np.eye(3) - projector) + cls.epsilon_e_0 * projector

    @classmethod
    def u(cls, z: complex | float) -> complex | float:
        return cls.u0 + cls.gradient * z

    @classmethod
    def scale(cls, z: complex | float) -> complex | float:
        return 1.0 / cls.u(z) ** 2

    @classmethod
    def metric(cls, z: complex | float) -> Array:
        return cls.scale(z) * cls.base_metric()

    @classmethod
    def dielectric(cls, z: complex | float) -> Array:
        return cls.scale(z) * cls.base_dielectric()

    @classmethod
    def analytic_position(
        cls, z_values: Array, q0: Array, z0: float, momentum: Array
    ) -> Array:
        base = cls.base_metric()
        a0 = base[:2, :2]
        b0 = base[:2, 2]
        c0 = float(base[2, 2])
        m0 = np.linalg.inv(a0)
        h0 = c0 - b0 @ m0 @ b0
        transverse_norm = float(momentum @ m0 @ momentum)
        u_initial = float(cls.u(z0))
        u_values = cls.u(z_values)
        if u_initial <= 0.0 or np.any(u_values <= 0.0):
            raise ValueError("the reciprocal profile requires chi(z)>0")
        radicand_initial = 1.0 - transverse_norm * u_initial * u_initial
        radicands = 1.0 - transverse_norm * u_values * u_values
        if radicand_initial <= 0.0 or np.min(radicands) <= 0.0:
            raise ValueError("the selected interval reaches the z turning point")

        # Rationalize the square-root difference and cancel gamma*Pi.
        # This expression also evaluates the continuous Pi=0 and gamma=0 limits.
        scalar = (
            np.sqrt(h0)
            * (z_values - z0)
            * (u_values + u_initial)
            / (np.sqrt(radicand_initial) + np.sqrt(radicands))
        )
        drift = -(z_values - z0)[:, None] * (m0 @ b0)[None, :]
        return q0 + scalar[:, None] * (m0 @ momentum)[None, :] + drift


class QuadraticTiltedUniaxial(FactorizedTiltedUniaxial):
    """A positive quadratic scale with its minimum inside the test interval."""

    minimum_scale = 0.9
    profile_rate = 0.23
    center_z = 2.3

    @classmethod
    def scale(cls, z: complex | float) -> complex | float:
        return cls.minimum_scale + cls.profile_rate**2 * (z - cls.center_z)**2

    @classmethod
    def analytic_position(
        cls, z_values: Array, q0: Array, z0: float, momentum: Array
    ) -> Array:
        base = cls.base_metric()
        m0 = np.linalg.inv(base[:2, :2])
        b0 = base[:2, 2]
        h0 = float(base[2, 2] - b0 @ m0 @ b0)
        transverse_norm = float(momentum @ m0 @ momentum)
        gap = cls.minimum_scale - transverse_norm
        if gap <= 0.0 or cls.profile_rate <= 0.0:
            raise ValueError("the quadratic closed form requires eta_c>Pi and lambda>0")
        scalar = (
            np.arcsinh(cls.profile_rate * (z_values - cls.center_z) / np.sqrt(gap))
            - np.arcsinh(cls.profile_rate * (z0 - cls.center_z) / np.sqrt(gap))
        ) / cls.profile_rate
        return (
            q0 + np.sqrt(h0) * scalar[:, None] * (m0 @ momentum)[None, :]
            - (z_values - z0)[:, None] * (m0 @ b0)[None, :]
        )


class IndefiniteTiltedUniaxial:
    """A factorized epsilon-negative uniaxial extraordinary metric."""

    epsilon_o_0 = -1.25
    epsilon_e_0 = 2.60
    scale_0 = 0.80
    gradient = 0.025
    axis = np.array([0.26, -0.19, 0.947])
    axis = axis / np.linalg.norm(axis)

    @classmethod
    def base_metric(cls) -> Array:
        projector = np.outer(cls.axis, cls.axis)
        return cls.epsilon_e_0 * (np.eye(3) - projector) + cls.epsilon_o_0 * projector

    @classmethod
    def base_dielectric(cls) -> Array:
        projector = np.outer(cls.axis, cls.axis)
        return cls.epsilon_o_0 * (np.eye(3) - projector) + cls.epsilon_e_0 * projector

    @classmethod
    def scale(cls, z: complex | float) -> complex | float:
        return cls.scale_0 + cls.gradient * z

    @classmethod
    def metric(cls, z: complex | float) -> Array:
        return cls.scale(z) * cls.base_metric()

    @classmethod
    def dielectric(cls, z: complex | float) -> Array:
        return cls.scale(z) * cls.base_dielectric()

    @classmethod
    def analytic_position(
        cls, z_values: Array, q0: Array, z0: float, momentum: Array
    ) -> Array:
        base = cls.base_metric()
        m0 = np.linalg.inv(base[:2, :2])
        b0 = base[:2, 2]
        h0 = float(base[2, 2] - b0 @ m0 @ b0)
        transverse_norm = float(momentum @ m0 @ momentum)
        scale_initial = float(cls.scale(z0))
        scale_values = cls.scale(z_values)
        if (
            h0 >= 0.0
            or scale_initial <= 0.0
            or np.min(scale_values) <= 0.0
            or scale_initial >= transverse_norm
            or np.max(scale_values) >= transverse_norm
        ):
            raise ValueError("the selected interval is not in the real indefinite domain")
        scalar = (
            2.0
            * np.sqrt(-h0)
            / cls.gradient
            * (np.sqrt(transverse_norm - scale_initial) - np.sqrt(transverse_norm - scale_values))
        )
        drift = -(z_values - z0)[:, None] * (m0 @ b0)[None, :]
        return q0 + scalar[:, None] * (m0 @ momentum)[None, :] + drift


class CoordinateShear:
    """A flat optical metric induced by X=x+f(z), Y=y+g(z)."""

    @staticmethod
    def f(z: complex | float) -> complex | float:
        return 0.31 * np.sin(0.52 * z) + 0.018 * z * z

    @staticmethod
    def g(z: complex | float) -> complex | float:
        return 0.24 * (1.0 - np.cos(0.41 * z)) - 0.027 * z

    @staticmethod
    def f_prime(z: complex | float) -> complex | float:
        return 0.31 * 0.52 * np.cos(0.52 * z) + 0.036 * z

    @staticmethod
    def g_prime(z: complex | float) -> complex | float:
        return 0.24 * 0.41 * np.sin(0.41 * z) - 0.027

    @classmethod
    def shift(cls, z: complex | float) -> Array:
        return np.array([cls.f(z), cls.g(z)])

    @classmethod
    def metric(cls, z: complex | float) -> Array:
        derivative = np.array([cls.f_prime(z), cls.g_prime(z)])
        dtype = np.result_type(z, float)
        result = np.eye(3, dtype=dtype)
        result[:2, 2] = derivative
        result[2, :2] = derivative
        result[2, 2] = 1.0 + derivative @ derivative
        return result

    @classmethod
    def analytic_position(
        cls, z_values: Array, q0: Array, z0: float, momentum: Array
    ) -> Array:
        if momentum @ momentum >= 1.0:
            raise ValueError("the coordinate-shear solution requires |p|<1")
        virtual_slope = momentum / np.sqrt(1.0 - momentum @ momentum)
        shift0 = cls.shift(z0)
        shifts = np.array([cls.shift(z) for z in z_values])
        return q0 + (z_values - z0)[:, None] * virtual_slope - (shifts - shift0)


def maxwell_residuals(metric, dielectric, z_values: Array, momentum: Array) -> dict[str, float]:
    normalized_singularity_residuals = []
    dispersion_residuals = []
    block_covector_errors = []
    poynting_direction_errors = []
    poynting_cosines = []
    normalized_axial_fluxes = []
    for z in z_values:
        metric_matrix = np.asarray(metric(float(z)), dtype=float)
        inverse_metric = np.linalg.inv(metric_matrix)
        alpha = inverse_metric[2, 2]
        beta = inverse_metric[2, :2] @ momentum
        gamma = momentum @ inverse_metric[:2, :2] @ momentum - 1.0
        discriminant = beta * beta - alpha * gamma
        if discriminant <= 0.0:
            raise ValueError("the selected Maxwell mode has no simple real root")
        vertical_momentum = (-beta + np.sqrt(discriminant)) / alpha
        covector = np.r_[momentum, vertical_momentum]
        # Independently compare the full inverse-metric root with the block formula.
        transverse_inverse = np.linalg.inv(metric_matrix[:2, :2])
        coupling = metric_matrix[:2, 2]
        schur = metric_matrix[2, 2] - coupling @ transverse_inverse @ coupling
        axial_factor = 1.0 - momentum @ transverse_inverse @ momentum
        block_vertical = (
            coupling @ transverse_inverse @ momentum
            + schur * np.sqrt(axial_factor / schur)
        )
        block_covector_errors.append(abs(vertical_momentum - block_vertical))
        maxwell_matrix = (
            (covector @ covector) * np.eye(3)
            - np.outer(covector, covector)
            - np.asarray(dielectric(float(z)), dtype=float)
        )
        _, singular_values, right_vectors = np.linalg.svd(maxwell_matrix)
        normalized_singularity_residuals.append(
            singular_values[-1] / singular_values[0]
        )
        dispersion_residuals.append(abs(covector @ inverse_metric @ covector - 1.0))
        # These closed-form dielectric examples have k0=mu=1.
        electric = right_vectors[-1]
        magnetic = np.cross(covector, electric)
        flux = np.cross(electric, magnetic)
        ray_direction = inverse_metric @ covector
        flux_unit = flux / np.linalg.norm(flux)
        ray_unit = ray_direction / np.linalg.norm(ray_direction)
        poynting_direction_errors.append(np.linalg.norm(np.cross(flux_unit, ray_unit)))
        poynting_cosines.append(flux_unit @ ray_unit)
        normalized_axial_fluxes.append(flux_unit[2])
    return {
        "maxwell_normalized_singularity_max_residual": float(
            max(normalized_singularity_residuals)
        ),
        "quadratic_dispersion_max_abs_residual": float(max(dispersion_residuals)),
        "full_inverse_vs_block_vertical_covector_max_abs_error": float(max(block_covector_errors)),
        "poynting_direction_max_sine_error": float(max(poynting_direction_errors)),
        "poynting_direction_min_cosine": float(min(poynting_cosines)),
        "minimum_normalized_axial_poynting_flux": float(min(normalized_axial_fluxes)),
    }


def heliconical_check() -> dict[str, float]:
    z0, z1 = 0.0, 8.0
    z_values = np.linspace(z0, z1, 161)
    q0 = np.array([0.13, -0.19])
    momentum = np.zeros(2)
    analytic = HeliconicalUniaxial.analytic_position(z_values, q0, z0)
    quadrature = gauss_position(HeliconicalUniaxial.metric, z_values, q0, z0, momentum)
    initial_slope = exact_velocity(HeliconicalUniaxial.metric, z0, momentum)
    states = _sample_geodesic(HeliconicalUniaxial.metric, z_values, q0, initial_slope)
    radius = abs(HeliconicalUniaxial.walkoff_coefficient() / HeliconicalUniaxial.twist_rate)
    center = np.array(
        [
            q0[0] + HeliconicalUniaxial.walkoff_coefficient()
            / HeliconicalUniaxial.twist_rate
            * np.sin(HeliconicalUniaxial.initial_azimuth),
            q0[1] - HeliconicalUniaxial.walkoff_coefficient()
            / HeliconicalUniaxial.twist_rate
            * np.cos(HeliconicalUniaxial.initial_azimuth),
        ]
    )
    circle_residual = np.max(np.abs(np.linalg.norm(analytic - center, axis=1) - radius))
    results = {
        "analytic_vs_general_quadrature_max_abs_error": float(np.max(np.abs(analytic - quadrature))),
        "analytic_vs_geodesic_position_max_abs_error": float(np.max(np.abs(analytic - states[:, :2]))),
        "circular_projection_max_abs_residual": float(circle_residual),
        "helix_radius": float(radius),
        "maximum_absolute_off_diagonal_metric_entry": float(
            max(
                np.max(np.abs(HeliconicalUniaxial.metric(z) - np.diag(np.diag(HeliconicalUniaxial.metric(z)))))
                for z in z_values
            )
        ),
    }
    results.update(
        maxwell_residuals(
            HeliconicalUniaxial.metric,
            HeliconicalUniaxial.dielectric,
            z_values,
            momentum,
        )
    )
    return results


def factorized_check() -> dict[str, float]:
    z0, z1 = 0.0, 8.0
    z_values = np.linspace(z0, z1, 161)
    q0 = np.array([-0.21, 0.14])
    momentum = np.array([0.54, -0.29])
    analytic = FactorizedTiltedUniaxial.analytic_position(z_values, q0, z0, momentum)
    quadrature = gauss_position(FactorizedTiltedUniaxial.metric, z_values, q0, z0, momentum)
    initial_slope = exact_velocity(FactorizedTiltedUniaxial.metric, z0, momentum)
    states = _sample_geodesic(FactorizedTiltedUniaxial.metric, z_values, q0, initial_slope)

    base = FactorizedTiltedUniaxial.base_metric()
    m0 = np.linalg.inv(base[:2, :2])
    transverse_norm = float(momentum @ m0 @ momentum)
    turning_u = 1.0 / np.sqrt(transverse_norm)
    turning_z = (turning_u - FactorizedTiltedUniaxial.u0) / FactorizedTiltedUniaxial.gradient
    results = {
        "analytic_vs_general_quadrature_max_abs_error": float(np.max(np.abs(analytic - quadrature))),
        "analytic_vs_geodesic_position_max_abs_error": float(np.max(np.abs(analytic - states[:, :2]))),
        "transverse_momentum_norm_P": transverse_norm,
        "predicted_turning_z": float(turning_z),
        "maximum_absolute_g12_g13_or_g23": float(
            np.max(np.abs(base - np.diag(np.diag(base))))
        ),
    }
    results.update(
        maxwell_residuals(
            FactorizedTiltedUniaxial.metric,
            FactorizedTiltedUniaxial.dielectric,
            z_values,
            momentum,
        )
    )
    return results



def quadratic_check() -> dict[str, float]:
    z0, z1 = 0.0, 8.0
    z_values = np.linspace(z0, z1, 161)
    q0 = np.array([-0.21, 0.14])
    momentum = np.array([0.54, -0.29])
    profile = QuadraticTiltedUniaxial
    analytic = profile.analytic_position(z_values, q0, z0, momentum)
    quadrature = gauss_position(profile.metric, z_values, q0, z0, momentum)
    states = _sample_geodesic(profile.metric, z_values, q0, exact_velocity(profile.metric, z0, momentum))

    # Solve the scalar inverse relation with quadrature, independently of the
    # asinh primitive used for the forward trajectory.
    base = profile.base_metric()
    a0, b0 = base[:2, :2], base[:2, 2]
    m0 = np.linalg.inv(a0)
    h0 = float(base[2, 2] - b0 @ m0 @ b0)
    displacement = analytic[-1] - q0
    shifted = displacement + (z1 - z0) * (m0 @ b0)
    target_norm = float(shifted @ a0 @ shifted)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    nodes = (z0 + z1) / 2.0 + (z1 - z0) / 2.0 * nodes
    weights = (z1 - z0) / 2.0 * weights

    def scalar_integral(pi):
        return float(weights @ (1.0 / np.sqrt(profile.scale(nodes) - pi)))

    low, high = 0.0, np.nextafter(profile.minimum_scale, 0.0)
    assert high * h0 * scalar_integral(high)**2 > target_norm
    for _ in range(80):
        middle = (low + high) / 2.0
        if middle * h0 * scalar_integral(middle)**2 < target_norm:
            low = middle
        else:
            high = middle
    pi = (low + high) / 2.0
    integral = scalar_integral(pi)
    recovered = (a0 @ displacement + b0 * (z1 - z0)) / (np.sqrt(h0) * integral)
    results = {
        "analytic_vs_general_quadrature_max_abs_error": float(np.max(np.abs(analytic - quadrature))),
        "analytic_vs_geodesic_position_max_abs_error": float(np.max(np.abs(analytic - states[:, :2]))),
        "scalar_endpoint_inverse_momentum_max_abs_error": float(np.max(np.abs(recovered - momentum))),
        "scalar_endpoint_inverse_Pi_abs_error": float(abs(pi - momentum @ m0 @ momentum)),
        "scalar_endpoint_equation_abs_residual": float(abs(pi * h0 * integral**2 - target_norm)),
        "scalar_endpoint_bisection_iterations": 80,
    }
    results.update(maxwell_residuals(profile.metric, profile.dielectric, z_values, momentum))
    return results

def indefinite_check() -> dict[str, float]:
    z0, z1 = 0.0, 8.0
    z_values = np.linspace(z0, z1, 161)
    q0 = np.array([-0.08, 0.12])
    momentum = np.array([1.80, -1.10])
    analytic = IndefiniteTiltedUniaxial.analytic_position(z_values, q0, z0, momentum)
    integral = gauss_position(IndefiniteTiltedUniaxial.metric, z_values, q0, z0, momentum)
    initial_slope = exact_velocity(IndefiniteTiltedUniaxial.metric, z0, momentum)
    states = _sample_geodesic(IndefiniteTiltedUniaxial.metric, z_values, q0, initial_slope)
    base = IndefiniteTiltedUniaxial.base_metric()
    m0 = np.linalg.inv(base[:2, :2])
    b0 = base[:2, 2]
    h0 = float(base[2, 2] - b0 @ m0 @ b0)
    transverse_norm = float(momentum @ m0 @ momentum)
    results = {
        "analytic_vs_general_quadrature_max_abs_error": float(np.max(np.abs(analytic - integral))),
        "analytic_vs_geodesic_position_max_abs_error": float(np.max(np.abs(analytic - states[:, :2]))),
        "schur_complement_h0": h0,
        "transverse_momentum_norm_P": transverse_norm,
        "maximum_scale": float(np.max(IndefiniteTiltedUniaxial.scale(z_values))),
        "minimum_transverse_block_eigenvalue": float(np.min(np.linalg.eigvalsh(base[:2, :2]))),
    }
    results.update(
        maxwell_residuals(
            IndefiniteTiltedUniaxial.metric,
            IndefiniteTiltedUniaxial.dielectric,
            z_values,
            momentum,
        )
    )
    return results


def coordinate_shear_check() -> dict[str, float]:
    z0, z1 = 0.0, 8.0
    z_values = np.linspace(z0, z1, 161)
    q0 = np.array([0.07, -0.11])
    momentum = np.array([0.34, -0.23])
    analytic = CoordinateShear.analytic_position(z_values, q0, z0, momentum)
    quadrature = gauss_position(CoordinateShear.metric, z_values, q0, z0, momentum)
    initial_slope = exact_velocity(CoordinateShear.metric, z0, momentum)
    states = _sample_geodesic(CoordinateShear.metric, z_values, q0, initial_slope)

    endpoint_shift = CoordinateShear.shift(z1) - CoordinateShear.shift(z0)
    virtual_displacement = analytic[-1] - q0 + endpoint_shift
    virtual_slope = virtual_displacement / (z1 - z0)
    recovered_momentum = virtual_slope / np.sqrt(1.0 + virtual_slope @ virtual_slope)
    determinant_residual = max(abs(np.linalg.det(CoordinateShear.metric(z)) - 1.0) for z in z_values)
    minimum_eigenvalue = min(np.linalg.eigvalsh(CoordinateShear.metric(z)).min() for z in z_values)
    return {
        "analytic_vs_general_quadrature_max_abs_error": float(np.max(np.abs(analytic - quadrature))),
        "analytic_vs_geodesic_position_max_abs_error": float(np.max(np.abs(analytic - states[:, :2]))),
        "explicit_endpoint_inverse_momentum_max_abs_error": float(
            np.max(np.abs(recovered_momentum - momentum))
        ),
        "metric_determinant_max_abs_error_from_one": float(determinant_residual),
        "minimum_metric_eigenvalue": float(minimum_eigenvalue),
    }



def reciprocal_stability_check() -> dict[str, object]:
    """Compare zero and small Pi/gamma with quadrature of the metric slope."""
    z0 = 0.3
    z_values = np.linspace(z0, 6.2, 17)
    q0 = np.array([-0.21, 0.14])
    cases = []
    for gradient in (0.0, 1.0e-14, 1.0e-10, 0.041, -0.041):
        profile = type("ReciprocalProfile", (FactorizedTiltedUniaxial,), {"gradient": gradient})
        for scale in (0.0, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-3, 1.0):
            momentum = scale * np.array([1.0, -0.5])
            analytic = profile.analytic_position(z_values, q0, z0, momentum)
            reference = gauss_position(profile.metric, z_values, q0, z0, momentum, order=128)
            error = float(np.max(np.abs(analytic - reference)))
            assert np.all(np.isfinite(analytic))
            assert error < 2.0e-14
            assert np.array_equal(analytic[0], q0)
            cases.append({"gradient": gradient, "momentum_scale": scale, "max_abs_error": error})
    # Exact conditions of the reported v1.0.0 cancellation failure.
    z_values = np.array([0.0, 8.0])
    momentum = np.array([1.0e-8, -0.5e-8])
    analytic = FactorizedTiltedUniaxial.analytic_position(z_values, np.zeros(2), 0.0, momentum)
    reference = gauss_position(
        FactorizedTiltedUniaxial.metric, z_values, np.zeros(2), 0.0, momentum, order=128
    )
    regression_error = float(np.max(np.abs(analytic - reference)))
    assert regression_error < 2.0e-14
    for gradient, momentum in ((-0.2, np.zeros(2)), (0.041, np.array([3.0, 0.0]))):
        profile = type("InvalidReciprocalProfile", (FactorizedTiltedUniaxial,), {"gradient": gradient})
        try:
            profile.analytic_position(z_values, np.zeros(2), 0.0, momentum)
        except ValueError:
            pass
        else:
            raise AssertionError("an inadmissible reciprocal profile was accepted")
    return {
        "cases": cases,
        "maximum_quadrature_error": max(case["max_abs_error"] for case in cases),
        "small_momentum_regression_error": regression_error,
        "invalid_profile_cases_rejected": 2,
    }

def run_checks() -> dict[str, object]:
    results = {
        "heliconical_uniaxial_extraordinary_ray": heliconical_check(),
        "factorized_tilted_uniaxial_grin": factorized_check(),
        "quadratic_tilted_uniaxial_grin": quadratic_check(),
        "indefinite_epsilon_negative_uniaxial": indefinite_check(),
        "coordinate_shear_transformation_medium": coordinate_shear_check(),
    }
    for diagnostics in results.values():
        assert diagnostics["analytic_vs_general_quadrature_max_abs_error"] < 1.0e-13
        assert diagnostics["analytic_vs_geodesic_position_max_abs_error"] < 1.0e-12
    assert results["coordinate_shear_transformation_medium"]["explicit_endpoint_inverse_momentum_max_abs_error"] < 1.0e-16
    for diagnostics in results.values():
        if "quadratic_dispersion_max_abs_residual" in diagnostics:
            assert diagnostics["quadratic_dispersion_max_abs_residual"] < 2.0e-13
            assert diagnostics["full_inverse_vs_block_vertical_covector_max_abs_error"] < 2.0e-13
            assert diagnostics["maxwell_normalized_singularity_max_residual"] < 1.0e-15
            assert diagnostics["poynting_direction_max_sine_error"] < 2.0e-13
            assert diagnostics["poynting_direction_min_cosine"] > 1.0 - 2.0e-13
            assert diagnostics["minimum_normalized_axial_poynting_flux"] > 0.0
    assert results["quadratic_tilted_uniaxial_grin"]["scalar_endpoint_inverse_momentum_max_abs_error"] < 1.0e-14
    assert results["quadratic_tilted_uniaxial_grin"]["scalar_endpoint_equation_abs_residual"] < 2.0e-13
    results["reciprocal_stability"] = reciprocal_stability_check()
    return results


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
