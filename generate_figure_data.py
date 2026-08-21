#!/usr/bin/env python3
"""Generate data for the combined main-text closed-form figure."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent

from verify_closed_forms import (  # noqa: E402
    FactorizedTiltedUniaxial,
    HeliconicalUniaxial,
    IndefiniteTiltedUniaxial,
)


def write_rows(filename: str, header: list[str], rows) -> None:
    target = PACKAGE_DIR / "data" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def heliconical_position(
    z_values: np.ndarray,
    cone_angle: float,
    twist_rate: float,
) -> np.ndarray:
    alpha = HeliconicalUniaxial.epsilon_e
    beta = HeliconicalUniaxial.epsilon_o
    delta = beta - alpha
    sine = np.sin(cone_angle)
    cosine = np.cos(cone_angle)
    walkoff = delta * sine * cosine / (alpha + delta * sine * sine)
    phase0 = HeliconicalUniaxial.initial_azimuth
    phase = twist_rate * z_values + phase0
    coefficient = walkoff / twist_rate
    return coefficient * np.column_stack(
        [np.sin(phase0) - np.sin(phase), np.cos(phase) - np.cos(phase0)]
    )


def main() -> None:
    z0 = 0.0
    z_values = np.linspace(z0, 8.0, 321)
    q0 = np.zeros(2)

    positive_momenta = (np.array([0.54, -0.29]), np.array([0.31, 0.24]))
    positive = [
        FactorizedTiltedUniaxial.analytic_position(z_values, q0, z0, momentum)
        for momentum in positive_momenta
    ]
    write_rows(
        "factorized_conditions.csv",
        ["z", "x1", "x2"],
        zip(z_values, positive[0][:, 0], positive[1][:, 0]),
    )

    indefinite_momenta = (np.array([1.80, -1.10]), np.array([1.60, 0.90]))
    indefinite = [
        IndefiniteTiltedUniaxial.analytic_position(z_values, q0, z0, momentum)
        for momentum in indefinite_momenta
    ]
    write_rows(
        "indefinite_conditions.csv",
        ["z", "x1", "x2"],
        zip(z_values, indefinite[0][:, 0], indefinite[1][:, 0]),
    )

    heliconical_parameters = ((0.48, 0.35), (0.61, 0.43), (0.72, 0.55))
    z_heliconical = np.linspace(z0, 2.0 * np.pi / heliconical_parameters[1][1], 401)
    helices = [
        heliconical_position(z_heliconical, angle, rate)
        for angle, rate in heliconical_parameters
    ]
    write_rows(
        "heliconical_conditions.csv",
        ["z", "x1", "y1", "x2", "y2", "x3", "y3"],
        zip(
            z_heliconical,
            helices[0][:, 0],
            helices[0][:, 1],
            helices[1][:, 0],
            helices[1][:, 1],
            helices[2][:, 0],
            helices[2][:, 1],
        ),
    )

    xi = (z_values - z0) / (z_values[-1] - z0)
    smoothstep = 3.0 * xi**2 - 2.0 * xi**3
    shifts = (0.30 * smoothstep, 0.60 * smoothstep, 0.90 * smoothstep)
    write_rows(
        "rectangular_shear_conditions.csv",
        ["z", "x1", "x2", "x3"],
        zip(z_values, shifts[0], shifts[1], shifts[2]),
    )


if __name__ == "__main__":
    main()
