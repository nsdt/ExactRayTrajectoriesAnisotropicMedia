#!/usr/bin/env python3
"""Numerical checks for the exact transverse integral in G = G(z).

The reference curve is evaluated from the conserved-momentum formula by
Gauss--Legendre integration.  It is compared with an adaptive embedded
Runge--Kutta--Fehlberg 4(5) integration of the non-affinely parameterized
three-dimensional geodesic equation.  SciPy is not required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss


Array = np.ndarray


def _rotation_x(angle: complex | float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.result_type(angle, float))


def _rotation_y(angle: complex | float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.result_type(angle, float))


def _rotation_z(angle: complex | float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.result_type(angle, float))


def test_tensor(z: complex | float) -> Array:
    """A smooth, positive-definite tensor with all off-diagonal entries."""
    diagonal = np.diag(
        [
            1.40 + 0.20 * np.sin(0.70 * z),
            0.90 + 0.10 * np.cos(0.40 * z),
            1.80 + 0.15 * np.sin(0.50 * z + 0.20),
        ]
    )
    rotation = (
        _rotation_z(0.25 * z)
        @ _rotation_y(0.18 * np.sin(0.60 * z))
        @ _rotation_x(0.12 * z)
    )
    return rotation @ diagonal @ rotation.T


def tensor_derivative(tensor, z: float) -> Array:
    """Complex-step derivative, used only by the geodesic integrator."""
    step = 1.0e-30
    return np.imag(tensor(z + 1j * step)) / step


def block_data(tensor, z: float) -> tuple[Array, Array, float]:
    metric = np.asarray(tensor(z), dtype=float)
    return metric[:2, :2], metric[:2, 2], float(metric[2, 2])


def initial_momentum(tensor, z0: float, slope0: Array) -> tuple[Array, float]:
    metric = np.asarray(tensor(z0), dtype=float)
    tangent = np.r_[slope0, 1.0]
    fermat = float(np.sqrt(tangent @ metric @ tangent))
    momentum = (metric[:2, :2] @ slope0 + metric[:2, 2]) / fermat
    return momentum, fermat


def exact_velocity(tensor, z: float, momentum: Array) -> Array:
    """dq/dz from the exact conserved-momentum formula."""
    a, b, c = block_data(tensor, z)
    a_inv = np.linalg.inv(a)
    schur = c - b @ a_inv @ b
    denominator = 1.0 - momentum @ a_inv @ momentum
    if schur / denominator <= 0.0:
        raise ValueError(f"non-real solution at z={z}: h={schur}, D={denominator}")
    fermat = np.sqrt(schur / denominator)
    return a_inv @ (fermat * momentum - b)


def gauss_position(tensor, z_values: Array, q0: Array, z0: float, momentum: Array, order: int = 64) -> Array:
    """Evaluate q(z) by independent Gauss--Legendre quadrature."""
    nodes, weights = leggauss(order)
    result = np.empty((len(z_values), 2), dtype=float)
    for index, z in enumerate(z_values):
        half = 0.5 * (z - z0)
        midpoint = 0.5 * (z + z0)
        integral = np.zeros(2)
        for node, weight in zip(nodes, weights):
            integral += weight * exact_velocity(tensor, midpoint + half * node, momentum)
        result[index] = q0 + half * integral
    return result


def endpoint_map_and_jacobian(
    tensor,
    z0: float,
    z1: float,
    momentum: Array,
    order: int = 64,
) -> tuple[Array, Array, float]:
    """Evaluate the endpoint displacement, its Jacobian, and sampled minimum D."""
    nodes, weights = leggauss(order)
    half = 0.5 * (z1 - z0)
    midpoint = 0.5 * (z1 + z0)
    displacement = np.zeros(2)
    jacobian = np.zeros((2, 2))
    minimum_denominator = np.inf

    for node, weight in zip(nodes, weights):
        z = midpoint + half * node
        a, b, c = block_data(tensor, z)
        a_inv = np.linalg.inv(a)
        schur = c - b @ a_inv @ b
        denominator = 1.0 - momentum @ a_inv @ momentum
        if schur / denominator <= 0.0:
            raise ValueError(f"non-real endpoint map at z={z}: h={schur}, D={denominator}")
        fermat = np.sqrt(schur / denominator)
        transformed_momentum = a_inv @ momentum
        displacement += weight * a_inv @ (fermat * momentum - b)
        jacobian += weight * fermat * (
            a_inv
            + np.outer(transformed_momentum, transformed_momentum) / denominator
        )
        minimum_denominator = min(minimum_denominator, denominator)

    return half * displacement, half * jacobian, float(minimum_denominator)


def invert_endpoint(
    tensor,
    z0: float,
    z1: float,
    target_displacement: Array,
    initial_momentum: Array,
    *,
    tolerance: float = 1.0e-13,
    max_iterations: int = 12,
) -> tuple[Array, int, float, float]:
    """Recover transverse momentum by Newton iteration on the endpoint map."""
    momentum = np.asarray(initial_momentum, dtype=float).copy()
    target = np.asarray(target_displacement, dtype=float)

    for evaluations in range(1, max_iterations + 1):
        displacement, jacobian, minimum_denominator = endpoint_map_and_jacobian(
            tensor, z0, z1, momentum
        )
        residual = displacement - target
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm < tolerance:
            return momentum, evaluations, residual_norm, minimum_denominator
        momentum -= np.linalg.solve(jacobian, residual)

    raise RuntimeError("endpoint Newton iteration did not converge")


def christoffel_symbols(tensor, z: float) -> Array:
    """Christoffel symbols of G(z), with coordinate order (x, y, z)."""
    metric = np.asarray(tensor(z), dtype=float)
    derivative = tensor_derivative(tensor, z)
    inverse = np.linalg.inv(metric)
    gamma = np.zeros((3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                total = 0.0
                for ell in range(3):
                    total += inverse[i, ell] * (
                        (derivative[ell, k] if j == 2 else 0.0)
                        + (derivative[ell, j] if k == 2 else 0.0)
                        - (derivative[j, k] if ell == 2 else 0.0)
                    )
                gamma[i, j, k] = 0.5 * total
    return gamma


def geodesic_rhs(tensor, z: float, state: Array) -> Array:
    """Geodesic equation with z used as a non-affine parameter."""
    slope = state[2:]
    tangent = np.r_[slope, 1.0]
    gamma = christoffel_symbols(tensor, z)
    contraction = np.einsum("ijk,j,k->i", gamma, tangent, tangent)
    curvature = -contraction[:2] + slope * contraction[2]
    return np.r_[slope, curvature]


def integrate_rkf45(
    rhs,
    t0: float,
    t1: float,
    state0: Array,
    t_eval: Array,
    *,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    initial_step: float | None = None,
    max_steps: int = 200_000,
) -> tuple[Array, Array, dict[str, int]]:
    """Integrate an ODE with Fehlberg's embedded fourth/fifth-order pair.

    The fifth-order estimate is accepted.  The difference between the
    fourth- and fifth-order formulas controls the step size.  Every requested
    output point is reached exactly by shortening the final internal step.
    """
    evaluation = np.asarray(t_eval, dtype=float)
    if evaluation.ndim != 1 or len(evaluation) == 0:
        raise ValueError("t_eval must be a nonempty one-dimensional array")
    if abs(evaluation[0] - t0) > 16.0 * np.finfo(float).eps:
        raise ValueError("t_eval must start at t0")
    direction = 1.0 if t1 >= t0 else -1.0
    if np.any(direction * np.diff(evaluation) < 0.0):
        raise ValueError("t_eval must be ordered in the integration direction")
    if abs(evaluation[-1] - t1) > 16.0 * np.finfo(float).eps * max(1.0, abs(t1)):
        raise ValueError("t_eval must end at t1")

    state = np.asarray(state0, dtype=float).copy()
    states = np.empty((len(evaluation), len(state)), dtype=float)
    states[0] = state
    t = float(t0)
    span = abs(t1 - t0)
    step = direction * (initial_step if initial_step is not None else max(span / 100.0, 1.0e-6))
    accepted = rejected = evaluations = 0
    output_index = 1

    while output_index < len(evaluation):
        target = float(evaluation[output_index])
        while direction * (target - t) > 32.0 * np.finfo(float).eps * max(1.0, abs(target)):
            if accepted + rejected >= max_steps:
                raise RuntimeError("RKF45 exceeded the maximum number of internal steps")
            step = direction * min(abs(step), abs(target - t))

            k1 = rhs(t, state)
            k2 = rhs(t + step / 4.0, state + step * k1 / 4.0)
            k3 = rhs(
                t + 3.0 * step / 8.0,
                state + step * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0),
            )
            k4 = rhs(
                t + 12.0 * step / 13.0,
                state
                + step
                * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
            )
            k5 = rhs(
                t + step,
                state
                + step
                * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
            )
            k6 = rhs(
                t + step / 2.0,
                state
                + step
                * (
                    -8.0 * k1 / 27.0
                    + 2.0 * k2
                    - 3544.0 * k3 / 2565.0
                    + 1859.0 * k4 / 4104.0
                    - 11.0 * k5 / 40.0
                ),
            )
            evaluations += 6
            fourth = state + step * (
                25.0 * k1 / 216.0
                + 1408.0 * k3 / 2565.0
                + 2197.0 * k4 / 4104.0
                - k5 / 5.0
            )
            fifth = state + step * (
                16.0 * k1 / 135.0
                + 6656.0 * k3 / 12825.0
                + 28561.0 * k4 / 56430.0
                - 9.0 * k5 / 50.0
                + 2.0 * k6 / 55.0
            )
            scale = atol + rtol * np.maximum(np.abs(state), np.abs(fifth))
            error_norm = float(np.max(np.abs(fifth - fourth) / scale))
            factor = 5.0 if error_norm == 0.0 else float(
                np.clip(0.9 * error_norm ** (-0.2), 0.2, 5.0)
            )

            if error_norm <= 1.0:
                t += step
                state = fifth
                accepted += 1
                step *= factor
            else:
                rejected += 1
                step *= min(1.0, factor)
                if abs(step) <= np.finfo(float).eps * max(1.0, abs(t)):
                    raise RuntimeError("RKF45 step underflow")

        t = target
        states[output_index] = state
        output_index += 1

    return evaluation.copy(), states, {
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "rhs_evaluations": evaluations,
    }


def integrate_geodesic_rkf45(
    tensor,
    z0: float,
    z1: float,
    state0: Array,
    z_eval: Array,
    *,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
) -> tuple[Array, Array, dict[str, int]]:
    return integrate_rkf45(
        lambda z, state: geodesic_rhs(tensor, z, state),
        z0,
        z1,
        state0,
        z_eval,
        rtol=rtol,
        atol=atol,
    )


def momentum_along_curve(tensor, z_values: Array, states: Array) -> Array:
    values = np.empty((len(z_values), 2), dtype=float)
    for index, (z, state) in enumerate(zip(z_values, states)):
        metric = np.asarray(tensor(z), dtype=float)
        tangent = np.r_[state[2:], 1.0]
        fermat = np.sqrt(tangent @ metric @ tangent)
        values[index] = (metric[:2, :2] @ state[2:] + metric[:2, 2]) / fermat
    return values


@dataclass(frozen=True)
class RotatingTilt:
    a: float = 1.25
    d: float = 0.85
    c: float = 2.10
    phi: float = 0.63
    kappa: float = 0.37

    @property
    def e(self) -> Array:
        return np.array([np.cos(self.phi), np.sin(self.phi)])

    def metric(self, z: complex | float) -> Array:
        theta = self.kappa * z
        e = self.e
        e_perp = np.array([-e[1], e[0]])
        u1 = np.r_[np.cos(theta) * e, np.sin(theta)]
        u2 = np.r_[e_perp, 0.0]
        u3 = np.r_[-np.sin(theta) * e, np.cos(theta)]
        return self.a * np.outer(u1, u1) + self.d * np.outer(u2, u2) + self.c * np.outer(u3, u3)

    def analytic_position(self, z_values: Array, q0: Array, z0: float) -> Array:
        def parallel_metric(z):
            theta = self.kappa * z
            return self.a * np.cos(theta) ** 2 + self.c * np.sin(theta) ** 2

        logarithm = np.log(np.array([parallel_metric(z) / parallel_metric(z0) for z in z_values]))
        return q0 + logarithm[:, None] * self.e[None, :] / (2.0 * self.kappa)


def diagonal_case_checks() -> dict[str, float]:
    z0 = 0.20
    z_values = np.linspace(z0, 4.70, 101)

    g1, g2 = 1.30, 0.75

    def metric_a(z):
        g3 = 1.60 + 0.25 * np.sin(0.70 * z)
        return np.diag([g1, g2, g3])

    slope0 = np.array([0.18, -0.11])
    momentum, _ = initial_momentum(metric_a, z0, slope0)
    computed = np.array([exact_velocity(metric_a, z, momentum) for z in z_values])
    g30 = metric_a(z0)[2, 2]
    expected = slope0[None, :] * np.sqrt(np.array([metric_a(z)[2, 2] / g30 for z in z_values]))[:, None]
    section_3a_error = float(np.max(np.abs(computed - expected)))

    sign = -1.0

    def metric_b(z):
        g = 1.10 + 0.16 * np.cos(0.45 * z)
        g3 = 1.70 + 0.20 * np.sin(0.55 * z)
        return np.diag([g, -g, g3])

    u0 = 0.14
    slope0 = np.array([u0, sign * u0])
    momentum, _ = initial_momentum(metric_b, z0, slope0)
    computed = np.array([exact_velocity(metric_b, z, momentum) for z in z_values])
    g10 = metric_b(z0)[0, 0]
    g30 = metric_b(z0)[2, 2]
    expected_x = np.array(
        [u0 * g10 * np.sqrt(metric_b(z)[2, 2]) / (np.sqrt(g30) * metric_b(z)[0, 0]) for z in z_values]
    )
    expected = np.column_stack([expected_x, sign * expected_x])
    section_3b_error = float(np.max(np.abs(computed - expected)))

    return {
        "2013_section_3A_velocity_max_abs_error": section_3a_error,
        "2013_section_3B_velocity_max_abs_error": section_3b_error,
    }


def run_checks() -> dict[str, object]:
    z0, z1 = 0.0, 8.0
    q0 = np.array([0.10, -0.20])
    slope0 = np.array([0.12, -0.08])
    state0 = np.r_[q0, slope0]
    momentum, fermat0 = initial_momentum(test_tensor, z0, slope0)
    reference_endpoint = gauss_position(test_tensor, np.array([z1]), q0, z0, momentum)[0]

    target_endpoint = np.array([0.9, -0.6])
    recovered_momentum, endpoint_evaluations, endpoint_residual, endpoint_minimum_d = invert_endpoint(
        test_tensor,
        z0,
        z1,
        target_endpoint - q0,
        np.zeros(2),
    )

    convergence: dict[str, object] = {}
    for tolerance in (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12):
        _, states, statistics = integrate_geodesic_rkf45(
            test_tensor,
            z0,
            z1,
            state0,
            np.array([z0, z1]),
            rtol=tolerance,
            atol=0.01 * tolerance,
        )
        convergence[f"{tolerance:.0e}"] = {
            "endpoint_inf_error": float(
                np.linalg.norm(states[-1, :2] - reference_endpoint, ord=np.inf)
            ),
            **statistics,
        }

    sampled_z = np.linspace(z0, z1, 81)
    finest_z, finest_states, finest_statistics = integrate_geodesic_rkf45(
        test_tensor,
        z0,
        z1,
        state0,
        sampled_z,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    reference_positions = gauss_position(test_tensor, sampled_z, q0, z0, momentum)
    reference_slopes = np.array([exact_velocity(test_tensor, z, momentum) for z in sampled_z])
    position_error = float(np.max(np.abs(finest_states[:, :2] - reference_positions)))
    slope_error = float(np.max(np.abs(finest_states[:, 2:] - reference_slopes)))
    numerical_momentum = momentum_along_curve(test_tensor, sampled_z, finest_states)
    momentum_error = float(np.max(np.abs(numerical_momentum - momentum[None, :])))

    tilt = RotatingTilt()
    tilt_z0 = 0.0
    tilt_z = np.linspace(tilt_z0, 5.0, 101)
    tilt_q0 = np.array([-0.15, 0.08])
    zero_momentum = np.zeros(2)
    tilt_quadrature = gauss_position(tilt.metric, tilt_z, tilt_q0, tilt_z0, zero_momentum)
    tilt_analytic = tilt.analytic_position(tilt_z, tilt_q0, tilt_z0)
    tilt_error = float(np.max(np.abs(tilt_quadrature - tilt_analytic)))

    results: dict[str, object] = {
        "full_nondiagonal_spd_case": {
            "initial_F": fermat0,
            "constant_momentum": momentum.tolist(),
            "minimum_metric_eigenvalue_on_samples": float(
                min(np.linalg.eigvalsh(test_tensor(z)).min() for z in sampled_z)
            ),
            "rkf45_endpoint_results_by_tolerance": convergence,
            "sampled_position_max_abs_error": position_error,
            "sampled_slope_max_abs_error": slope_error,
            "sampled_momentum_max_abs_drift": momentum_error,
            "sampled_rkf45_statistics": finest_statistics,
            "endpoint_inversion": {
                "target_endpoint": target_endpoint.tolist(),
                "recovered_momentum": recovered_momentum.tolist(),
                "endpoint_map_evaluations": endpoint_evaluations,
                "endpoint_inf_residual": endpoint_residual,
                "minimum_D_at_gauss_nodes": endpoint_minimum_d,
            },
        },
        "rotating_tilt_with_g13_g23": {
            "quadrature_vs_logarithmic_solution_max_abs_error": tilt_error,
        },
        "diagonal_reductions": diagonal_case_checks(),
    }

    assert convergence["1e-12"]["endpoint_inf_error"] < 2.0e-10
    assert position_error < 2.0e-10
    assert slope_error < 2.0e-10
    assert momentum_error < 2.0e-10
    assert endpoint_evaluations == 5
    assert endpoint_residual < 1.0e-12
    assert endpoint_minimum_d > 0.88
    assert tilt_error < 2.0e-12
    assert max(results["diagonal_reductions"].values()) < 2.0e-13
    return results


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
