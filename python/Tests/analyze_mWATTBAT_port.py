#!/usr/bin/env python3
"""Check the mWATTBAT lumped-port geometry and electrical size."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import gdspy
import numpy as np


C0 = 299_792_458.0
AXES = ("x", "y", "z")


def parse_vector(element: ET.Element, prefix: str) -> np.ndarray:
    return np.array(
        [float(element.attrib[f"{prefix}{axis.upper()}"]) for axis in AXES],
        dtype=float,
    )


def read_port_box(simulation_xml: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    root = ET.parse(simulation_xml).getroot()
    lumped = root.find(".//LumpedElement[@Name='port_resist_1']")
    if lumped is None:
        raise ValueError(f"port_resist_1 not found in {simulation_xml}")
    box = lumped.find("./Primitives/Box")
    if box is None:
        raise ValueError(f"port_resist_1 has no box in {simulation_xml}")
    points = box.findall("./P1") + box.findall("./P2")
    if len(points) != 2:
        raise ValueError(f"Unexpected port box in {simulation_xml}")
    start = parse_vector(points[0], "")
    stop = parse_vector(points[1], "")

    meshes = {}
    for axis in AXES:
        element = root.find(f".//{axis.upper()}Lines")
        if element is None or element.text is None:
            raise ValueError(f"{axis.upper()}Lines not found in {simulation_xml}")
        meshes[axis] = np.fromstring(element.text, sep=",")
    return start, stop, meshes


def read_port_marker(gds_path: Path, layer: int, datatype: int) -> tuple[np.ndarray, np.ndarray]:
    library = gdspy.GdsLibrary(infile=str(gds_path))
    cells = library.top_level()
    cell = max(cells, key=lambda candidate: int(layer in candidate.get_layers()))
    cell.flatten(single_layer=None, single_datatype=None, single_texttype=None)
    polygons = []
    for (poly_layer, poly_datatype), layer_polygons in cell.get_polygons(
        by_spec=True, depth=0
    ).items():
        if int(poly_layer) == layer and int(poly_datatype) == datatype:
            polygons.extend(layer_polygons)
    if not polygons:
        raise ValueError(f"No GDS polygons found on layer {layer}/{datatype}")
    points = np.concatenate(polygons, axis=0)
    return np.min(points, axis=0), np.max(points, axis=0)


def read_permittivities(stackup: Path) -> dict[str, float]:
    root = ET.parse(stackup).getroot()
    values = {}
    for material in root.findall(".//Material"):
        name = material.attrib.get("Name", "")
        if "Permittivity" in material.attrib:
            values[name] = float(material.attrib["Permittivity"])
    return values


def wavelength_um(frequency: float, permittivity: float) -> float:
    return C0 / (frequency * math.sqrt(permittivity)) * 1e6


def report(
    simulation_xml: Path,
    gds_path: Path,
    stackup: Path,
    ftarget: float,
    fmax: float,
) -> None:
    start, stop, meshes = read_port_box(simulation_xml)
    marker_start, marker_stop = read_port_marker(gds_path, layer=201, datatype=0)
    dimensions = np.abs(stop - start)
    marker_dimensions = np.abs(marker_stop - marker_start)
    direction = int(ET.parse(simulation_xml).getroot().find(
        ".//LumpedElement[@Name='port_resist_1']"
    ).attrib["Direction"])

    print(f"Simulation XML: {simulation_xml}")
    print(f"GDS port marker bbox (um): {marker_start} -> {marker_stop}")
    print(f"Generated lumped-port box (um): {start} -> {stop}")
    print(f"Generated dimensions (um): {dimensions}")
    print(f"GDS marker dimensions (um): {marker_dimensions}")
    print(f"Port direction: {AXES[direction]} (axis index {direction})")

    for axis, length in zip(AXES, dimensions):
        line_values = meshes[axis]
        axis_index = AXES.index(axis)
        lower, upper = sorted((start[axis_index], stop[axis_index]))
        inside = line_values[(line_values >= lower - 1e-9) & (line_values <= upper + 1e-9)]
        cells = max(0, len(inside) - 1) if len(inside) else 0
        print(f"  {axis}: {length:.6g} um, {cells} mesh cells in port span")

    permittivities = read_permittivities(stackup)
    print("Electrical size of the longest port dimension:")
    for label, eps in (
        ("air", 1.0),
        ("SiO2", permittivities.get("SiO2", 4.1)),
        ("substrate", permittivities.get("Substrate", 11.9)),
    ):
        for frequency, frequency_label in ((ftarget, "target"), (fmax, "band edge")):
            wavelength = wavelength_um(frequency, eps)
            ratio = dimensions.max() / wavelength
            print(
                f"  {label:9s} {frequency_label:8s}: "
                f"lambda={wavelength:8.2f} um, max_dim/lambda={ratio:.4f}"
            )

    print("Assessment: concentrated-port guidance requires the complete box to be much smaller than lambda.")
    if dimensions.max() / wavelength_um(fmax, permittivities.get("SiO2", 4.1)) > 0.1:
        print("WARNING: the generated lumped port is not electrically small in the SiO2 environment.")


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_xml", type=Path)
    parser.add_argument("--gds", type=Path, default=script_directory / "mWATTBAT_openEMS.gds")
    parser.add_argument("--stackup", type=Path, default=script_directory / "SG13G2_200um.xml")
    parser.add_argument("--target-frequency", type=float, default=150e9)
    parser.add_argument("--maximum-frequency", type=float, default=200e9)
    args = parser.parse_args()
    report(args.simulation_xml, args.gds, args.stackup, args.target_frequency, args.maximum_frequency)


if __name__ == "__main__":
    main()
