#!/usr/bin/env python3
"""Generate reproducible Kratos cutting-case JSON files from centralized parameters."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "cases" / "cutting2d" / "case_parameters.json"


def _load_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_project_parameters(base: dict, case_name: str, scale: dict) -> dict:
    kinematics = base["kinematics"]
    geometry = base["geometry"]
    contact = base["contact"]
    remeshing = base["remeshing"]

    dt = kinematics["time_step_s"] * scale["time_step_scale"]
    end_time = kinematics["end_time_s"] * scale["end_time_scale"]

    return {
        "problem_data": {
            "problem_name": f"cutting2d_{case_name}",
            "parallel_type": "OpenMP",
            "domain_size": geometry["domain_size"],
            "start_time": 0.0,
            "end_time": end_time,
            "time_step": dt,
            "echo_level": 1,
            "analysis_type": kinematics["analysis_type"]
        },
        "solver_settings": {
            "solver_type": "solid_mechanics_solver",
            "model_part_name": "Solid Domain",
            "domain_size": geometry["domain_size"],
            "echo_level": 1,
            "buffer_size": kinematics["buffer_size"],
            "time_integration_method": "Implicit",
            "line_search": True,
            "max_iteration": 30,
            "compute_reactions": True,
            "reform_dofs_at_each_step": True
        },
        "processes": {
            "constraints_process_list": [
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "Parameters": {
                        "model_part_name": "Tool",
                        "variable_name": "VELOCITY",
                        "interval": [0.0, "End"],
                        "value": [kinematics["tool_velocity_m_per_s"], 0.0, 0.0],
                        "constrained": [True, True, True]
                    }
                }
            ],
            "contact_process_list": [
                {
                    "python_module": "contact_domain_process",
                    "kratos_module": "KratosMultiphysics.ContactMechanicsApplication",
                    "Parameters": {
                        "friction_active": True,
                        "friction_coefficient": contact["coulomb_friction"],
                        "penalty_parameter": contact["penalty_parameter"],
                        "normal_tolerance": contact["normal_tolerance"],
                        "tangent_tolerance": contact["tangent_tolerance"]
                    }
                }
            ],
            "remeshing_process_list": [
                {
                    "python_module": "remesh_domains_process",
                    "kratos_module": "KratosMultiphysics.PfemSolidMechanicsApplication",
                    "Parameters": {
                        "model_part_name": "Solid Domain",
                        "meshing_frequency": 1,
                        "alpha_shape": remeshing["alpha_shape"],
                        "h_factor": remeshing["h_factor"],
                        "mesh_size_factor": scale["mesh_size_factor"],
                        "critical_plastic_strain": remeshing["critical_plastic_strain"],
                        "remesh": remeshing["enabled"]
                    }
                }
            ]
        },
        "output_configuration": {
            "result_file_configuration": {
                "gidpost_flags": {
                    "GiDPostMode": "GiD_PostBinary",
                    "WriteDeformedMeshFlag": "WriteDeformed",
                    "WriteConditionsFlag": "WriteConditions",
                    "MultiFileFlag": "SingleFile"
                },
                "output_interval": base["output"]["output_interval_steps"],
                "nodal_results": [
                    "DISPLACEMENT",
                    "VELOCITY",
                    "REACTION",
                    "CONTACT_FORCE",
                    "EQUIVALENT_PLASTIC_STRAIN"
                ],
                "gauss_point_results": [
                    "CAUCHY_STRESS_VECTOR",
                    "VON_MISES_STRESS"
                ]
            }
        },
        "model_description": {
            "orthogonal_cutting_plane_strain": True,
            "workpiece_length_m": geometry["workpiece_length_m"],
            "workpiece_height_m": geometry["workpiece_height_m"],
            "undeformed_chip_thickness_m": geometry["undeformed_chip_thickness_m"],
            "tool_rake_angle_deg": geometry["tool_rake_angle_deg"],
            "tool_clearance_angle_deg": geometry["tool_clearance_angle_deg"],
            "tool_edge_radius_m": geometry["tool_edge_radius_m"],
            "cutting_depth_m": geometry["cutting_depth_m"],
            "initial_tool_x_m": geometry["initial_tool_x_m"],
            "initial_tool_y_m": geometry["initial_tool_y_m"]
        }
    }


def _build_materials(base: dict) -> dict:
    m = base["material"]
    return {
        "properties": [
            {
                "model_part_name": "WorkPiece",
                "properties_id": 1,
                "Material": {
                    "constitutive_law": {
                        "name": "HyperElasticPlasticJ2PlaneStrain2DLaw"
                    },
                    "Variables": {
                        "DENSITY": m["density_kg_m3"],
                        "YOUNG_MODULUS": m["young_modulus_pa"],
                        "POISSON_RATIO": m["poisson_ratio"],
                        "YIELD_STRESS": m["yield_stress_pa"],
                        "KINEMATIC_HARDENING_MODULUS": m["kinematic_hardening_modulus_pa"],
                        "HARDENING_EXPONENT": m["hardening_exponent"],
                        "REFERENCE_HARDENING_MODULUS": m["reference_hardening_modulus_pa"],
                        "INFINITY_HARDENING_MODULUS": m["infinity_hardening_modulus_pa"]
                    },
                    "Tables": {}
                }
            }
        ]
    }


def _write_case(case_dir: Path, project_parameters: dict, materials: dict, base: dict) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "ProjectParameters.json").open("w", encoding="utf-8") as f:
        json.dump(project_parameters, f, indent=2)
        f.write("\n")

    with (case_dir / "Materials.json").open("w", encoding="utf-8") as f:
        json.dump(materials, f, indent=2)
        f.write("\n")

    with (case_dir / "CaseMetadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "kratos_commit": base["kratos"]["commit"],
                "kratos_version": base["kratos"]["version"],
                "required_applications": base["kratos"]["required_applications"],
                "notes": "Mesh generation and run are scripted from run_case.sh."
            },
            f,
            indent=2,
        )
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameters",
        type=Path,
        default=PARAMS_PATH,
        help="Path to case_parameters.json",
    )
    args = parser.parse_args()

    base = _load_params(args.parameters)
    materials = _build_materials(base)

    for case_name, scale in base["cases"].items():
        project_parameters = _build_project_parameters(base, case_name, scale)
        case_dir = ROOT / "cases" / "cutting2d" / case_name
        _write_case(case_dir, project_parameters, deepcopy(materials), base)


if __name__ == "__main__":
    main()
