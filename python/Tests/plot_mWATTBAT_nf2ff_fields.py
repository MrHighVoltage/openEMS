#!/usr/bin/env python3
"""Plot complex NF2FF face fields exported by an openEMS simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FACE_NAMES = ("x-min", "x-max", "y-min", "y-max", "z-min", "z-max")


def resolve_subdirectory(path: Path) -> Path:
    """Accept either a simulation data directory or its sub-1 directory."""
    if (path / "nf2ff_E_0.h5").exists():
        return path
    subdirectory = path / "sub-1"
    if (subdirectory / "nf2ff_E_0.h5").exists():
        return subdirectory
    raise FileNotFoundError(f"No NF2FF face files found below {path}")


def read_face(path: Path, frequency_index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        real = np.asarray(handle["FieldData/FD/f0_real"])
        imag = np.asarray(handle["FieldData/FD/f0_imag"])
        coordinates = {
            axis: np.asarray(handle[f"Mesh/{axis}"]) for axis in ("x", "y", "z")
        }

    if real.shape != imag.shape or real.ndim not in (4, 5):
        raise ValueError(f"Unexpected field shape in {path}: {real.shape}, {imag.shape}")
    if real.ndim == 4:
        if frequency_index != 0:
            raise IndexError(f"Only frequency index 0 is present in {path}")
        complex_field = real + 1j * imag
    else:
        if not 0 <= frequency_index < real.shape[-1]:
            raise IndexError(
                f"Frequency index {frequency_index} is outside [0, {real.shape[-1]}) in {path}"
            )
        complex_field = real[..., frequency_index] + 1j * imag[..., frequency_index]

    # openEMS stores the spatial dimensions in z, y, x order after the
    # three vector components. Convert each face to a 2-D tangent-plane map.
    magnitude = np.sqrt(np.sum(np.abs(complex_field) ** 2, axis=0))
    normal_axis = next(
        axis for axis in ("x", "y", "z") if coordinates[axis].size == 1
    )

    if normal_axis == "x":
        values = magnitude[:, :, 0]
        horizontal, vertical = "y", "z"
    elif normal_axis == "y":
        values = magnitude[:, 0, :]
        horizontal, vertical = "x", "z"
    else:
        values = magnitude[0, :, :]
        horizontal, vertical = "x", "y"

    expected_shape = (coordinates[vertical].size, coordinates[horizontal].size)
    if values.shape != expected_shape:
        raise ValueError(
            f"Cannot map {path}: values {values.shape}, expected {expected_shape}"
        )
    return values, {
        "horizontal": coordinates[horizontal],
        "vertical": coordinates[vertical],
    }


def relative_db(values: np.ndarray, reference: float, floor_db: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_value = reference * 10.0 ** (floor_db / 20.0)
        result = 20.0 * np.log10(np.maximum(values, floor_value) / reference)
    return np.maximum(result, floor_db)


def plot_fields(data_directory: Path, output: Path, frequency_index: int, floor_db: float) -> None:
    fields: dict[str, list[np.ndarray]] = {"E": [], "H": []}
    metadata: list[dict[str, np.ndarray]] = []

    for face_index in range(6):
        for kind in fields:
            values, face_metadata = read_face(
                data_directory / f"nf2ff_{kind}_{face_index}.h5", frequency_index
            )
            fields[kind].append(values)
        metadata.append(face_metadata)

    references = {
        kind: max(float(np.max(values)) for values in values_list)
        for kind, values_list in fields.items()
    }
    if any(reference <= 0.0 or not np.isfinite(reference) for reference in references.values()):
        raise ValueError(f"NF2FF field data is empty or non-finite: {references}")

    figure, axes = plt.subplots(2, 6, figsize=(18, 6.8), constrained_layout=True)
    mesh = None
    for row, kind in enumerate(("E", "H")):
        for face_index, axis in enumerate(axes[row]):
            values = relative_db(fields[kind][face_index], references[kind], floor_db)
            face_metadata = metadata[face_index]
            horizontal = face_metadata["horizontal"] * 1e6
            vertical = face_metadata["vertical"] * 1e6
            mesh = axis.pcolormesh(
                horizontal,
                vertical,
                values,
                shading="auto",
                cmap="viridis",
                vmin=floor_db,
                vmax=0.0,
            )
            axis.set_title(f"{kind}, {FACE_NAMES[face_index]}")
            axis.set_xlabel("tangent coordinate (um)")
            axis.set_ylabel("tangent coordinate (um)")
            axis.set_aspect("auto")

    figure.suptitle(
        "mWATTBAT NF2FF face fields, frequency index "
        f"{frequency_index} (relative magnitude, dB)\n"
        f"E max={references['E']:.3e}, H max={references['H']:.3e}",
    )
    if mesh is not None:
        figure.colorbar(mesh, ax=axes, label="magnitude relative to row maximum (dB)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print(f"Wrote {output}")
    for kind, values_list in fields.items():
        print(
            f"{kind}: global max={references[kind]:.6e}, "
            f"face maxima={[f'{np.max(values):.6e}' for values in values_list]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: <data directory>/sub-1/nf2ff_fields.png)",
    )
    parser.add_argument("--frequency-index", type=int, default=0)
    parser.add_argument("--floor-db", type=float, default=-60.0)
    args = parser.parse_args()

    subdirectory = resolve_subdirectory(args.data_directory)
    output = args.output or subdirectory / "nf2ff_fields.png"
    plot_fields(subdirectory, output, args.frequency_index, args.floor_db)


if __name__ == "__main__":
    main()
