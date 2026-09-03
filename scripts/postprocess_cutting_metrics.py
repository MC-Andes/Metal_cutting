#!/usr/bin/env python3
"""Compute cutting metrics from CSV outputs (forces, energies, chip geometry)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_rows(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forces", type=Path, required=True)
    parser.add_argument("--energies", type=Path, required=True)
    parser.add_argument("--chip", type=Path, required=True)
    parser.add_argument("--transient-cutoff-time", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    forces = [r for r in _load_rows(args.forces) if r["time"] >= args.transient_cutoff_time]
    energies = [r for r in _load_rows(args.energies) if r["time"] >= args.transient_cutoff_time]
    chip_rows = _load_rows(args.chip)

    chip_thickness = chip_rows[-1]["chip_thickness"]
    undeformed_thickness = chip_rows[-1]["undeformed_thickness"]

    summary = {
        "time_window_start": args.transient_cutoff_time,
        "mean_cutting_force": _mean([r["cutting_force"] for r in forces]),
        "mean_thrust_force": _mean([r["thrust_force"] for r in forces]),
        "mean_resultant_force": _mean([r["resultant_force"] for r in forces]),
        "apparent_friction_coeff": _mean([r["apparent_mu"] for r in forces]),
        "chip_thickness": chip_thickness,
        "chip_compression_ratio": chip_thickness / undeformed_thickness if undeformed_thickness else 0.0,
        "final_internal_energy": energies[-1]["internal_energy"] if energies else 0.0,
        "final_kinetic_energy": energies[-1]["kinetic_energy"] if energies else 0.0,
        "final_external_work": energies[-1]["external_work"] if energies else 0.0,
        "final_contact_dissipation": energies[-1]["contact_dissipation"] if energies else 0.0,
        "final_plastic_dissipation": energies[-1]["plastic_dissipation"] if energies else 0.0
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
