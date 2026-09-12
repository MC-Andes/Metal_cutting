"""Sourced cards, analytic finite rounded tools and independently refinable clouds."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction
import copy
import json
import math
from pathlib import Path

import numpy as np

from .config import (Acceptance, Boundary, CaseConfig, Geometry, MATERIAL_UNITS,
                     MATERIAL_VERSION, Output, Solver, Thermal, VERSION, EXPLICIT_VERSION,REGISTRY_VERSION,REGISTRY_BACKEND, load_case)
from scripts.klippel_material_cards import select_card
from scripts.rounded_tool_geometry import RoundedTool, tool_vertices


_CONVENTION = {
    "kinematics": "3D finite Hencky J2 under Fzz=1 plane strain; Fp remains 3D",
    "stress": "JC resistance is Cauchy; local return uses q_tau=J*Y",
    "rate_factor": "1+C*log(max(1,alpha_dot/reference_rate))",
    "thermal_factor": "1-clamp((T-Tref)/(Tm-Tref),0,1)**m; independent of beta",
    "beta": "fraction of plastic work converted to heat; beta=0 retains thermal softening",
    "solid_range": "Stop before T>=Tm; no phase change or physical melt model",
    "properties": "constant density reference, E, nu, cp, k and beta; no thermal expansion",
}


def material_card(name: str) -> dict:
    """Return new data only. SyntheticJC is a numerical control, never an alloy."""
    if name.lower().replace("_", "").replace("-", "") in {"synthetic", "syntheticjc"}:
        properties = {
            "name": "SyntheticJC", "density_kg_m3": 2700., "young_modulus_pa": 70e9,
            "poisson_ratio": .3, "initial_temperature_k": 293.15,
            "jc_a_pa": 100e6, "jc_b_pa": 200e6, "jc_c": .01, "jc_n": .3,
            "jc_m": 1., "reference_strain_rate_s_1": 1.,
            "reference_temperature_k": 293.15, "melt_temperature_k": 1000.,
            "specific_heat_j_kg_k": 900., "thermal_conductivity_w_m_k": 100.,
            "taylor_quinney": .9,
        }
        evidence = {k: {"value": v, "unit": MATERIAL_UNITS[k], "status": "synthetic",
                       "source": None, "source_locator": "cutting2d.cases.material_card",
                       "assumptions": ["Arbitrary numerical control; does not represent any measured alloy"],
                       "uncertainty": None} for k, v in properties.items() if k != "name"}
        identifier, status = "synthetic_jc_v1", "synthetic_control"
        limitations = ["Synthetic numerical control; no experimental validity or uncertainty distribution"]
    else:
        old = select_card(name)
        properties = old["material"]
        evidence = copy.deepcopy(old["provenance"]["fields"])
        identifier, status = old["card_id"] + "_finite_v1", "ranges_not_established"
        limitations = [value for value in old["provenance"]["limitations"]
                       if not any(x in value for x in ("Kratos", "metadata; this card"))]
        limitations += [
            "Migrated numeric constants unchanged; active finite law is described by constitutive_convention",
            "Experimental plastic strain/rate applicability intervals have not been established here",
            "Tref and Tm delimit the implemented thermal normalization, not an experimental validity interval",
            "Friction is a pair/contact assumption in the case thermal section, not a material constant",
        ]
    return {"schema_version": MATERIAL_VERSION, "id": identifier, "law": "finite_jc",
            "properties": properties,
            "provenance": {"fields": evidence, "limitations": limitations,
                           "constitutive_convention": copy.deepcopy(_CONVENTION)},
            "validity": {"status": status, "temperature_k": [properties["reference_temperature_k"], properties["melt_temperature_k"]],
                         "experimental_plastic_strain": None, "experimental_strain_rate_s_1": None}}


def make_tool(config: CaseConfig, time: float = 0.) -> RoundedTool:
    """Exact same geometry for runtime/export; x advances, lowest y gives depth.

    initial_gap is the horizontal gap from the *physical* rightmost rounded
    boundary to the x=0 stock boundary, not the virtual sharp intersection.
    Tool face lengths are deterministic geometry extents independent of h/ppc.
    """
    if not math.isfinite(time) or time < 0:
        raise ValueError("Tool time must be finite and nonnegative")
    g = config.geometry
    face_length = 3 * max(g.length, g.height, g.edge_radius)
    settings = {"tool_initial_x_m": 0., "tool_edge_y_m": 0.,
                "tool_velocity_m_s": g.tool_speed, "tool_rake_angle_deg": g.rake_angle_deg,
                "tool_clearance_angle_deg": g.clearance_angle_deg,
                "tool_rake_length_m": face_length, "tool_clearance_length_m": face_length}
    reference = RoundedTool(tool_vertices(settings), g.edge_radius)
    candidates = [*reference.vertices[1:], *reference.tangent_points]
    if reference._arc_parameter((1., 0.)) <= reference.arc_sweep:
        candidates.append((reference.center[0] + reference.radius, reference.center[1]))
    front_x = max(p[0] for p in candidates)
    settings["tool_initial_x_m"] = -g.initial_gap - front_x
    settings["tool_edge_y_m"] = g.height - g.uncut_thickness - reference.lowest_point_offset
    return RoundedTool(tool_vertices(settings, time), g.edge_radius)


def _subintervals(length: float, h: float, particles_per_axis: int) -> list[tuple[float, float]]:
    """Clip only the last background cell; subdivide every occupied interval."""
    count = int((Decimal(str(length)) / Decimal(str(h))).to_integral_value(rounding=ROUND_CEILING))
    # Unit arithmetic such as 25*1e-6 can put h one ulp below length/4.
    # Absorb only a terminal interval indistinguishable at the domain scale
    # into the preceding cell; its volume is retained through hi=length.
    if count > 1 and 0 < length-(count-1)*h <= 8*math.ulp(length):
        count -= 1
    intervals = []
    for i in range(count):
        lo = min(i * h, length)
        hi = length if i == count-1 else min((i + 1) * h, length)
        if hi <= lo:
            continue
        a, width = Fraction.from_float(lo), Fraction.from_float(hi) - Fraction.from_float(lo)
        for j in range(particles_per_axis):
            left = float(a + j * width / particles_per_axis)
            right = float(a + (j + 1) * width / particles_per_axis)
            if right <= left:
                raise ValueError("Particle quadrature under-resolved in Float64")
            intervals.append((left, right))
    return intervals


def generate_particles(config: CaseConfig) -> dict[str, np.ndarray]:
    """Midpoint quadrature of the complete rectangle, with V0=A0*physical width.

    Increasing h resolution and ppc are independent. Clipped outer subcells
    retain their actual area; no area/mass renormalization or missing particles.
    """
    g, s = config.geometry, config.solver
    xs, ys = _subintervals(g.length, s.h, s.particles_per_axis), _subintervals(g.height, s.h, s.particles_per_axis)
    n = len(xs) * len(ys)
    positions, half_sizes = np.empty((n, 2)), np.empty((n, 2))
    volumes, masses = np.empty(n), np.empty(n)
    width, rho = Fraction.from_float(g.width), Fraction.from_float(config.material["density_kg_m3"])
    index = 0
    for y0, y1 in ys:
        ylo, yhi = Fraction.from_float(y0), Fraction.from_float(y1)
        for x0, x1 in xs:
            xlo, xhi = Fraction.from_float(x0), Fraction.from_float(x1)
            positions[index] = (float((xlo + xhi) / 2), float((ylo + yhi) / 2))
            half_sizes[index] = (float((xhi - xlo) / 2), float((yhi - ylo) / 2))
            volume = float((xhi - xlo) * (yhi - ylo) * width)
            mass = float(Fraction.from_float(volume) * rho)
            if not math.isfinite(volume) or not math.isfinite(mass) or volume <= 0 or mass <= 0:
                raise ValueError("Particle volume/mass underflow or overflow")
            volumes[index], masses[index] = volume, mass
            index += 1
    return {"positions": positions, "reference_positions": positions.copy(),
            "reference_half_sizes": half_sizes, "volumes": volumes, "masses": masses,
            "ids": np.arange(1, n + 1, dtype=np.int64)}


def baseline_case(material: str = "Ck45", *, case_id: str | None = None) -> dict:
    """Plan §8.2 geometry. dt is an exploratory wave-speed bound, never selected."""
    card = material_card(material)
    mat = card["properties"]
    identifier = case_id or (mat["name"].lower() + "_basal")
    geometry = Geometry(.0015, .0006, .001, .0001, .000025, 10., 7., .000025, 2.5)
    h, cfl = .0000125, .2
    lame = mat["young_modulus_pa"] * mat["poisson_ratio"] / ((1 + mat["poisson_ratio"]) * (1 - 2 * mat["poisson_ratio"]))
    shear = mat["young_modulus_pa"] / (2 * (1 + mat["poisson_ratio"]))
    wave_speed = math.sqrt((lame + 2 * shear) / mat["density_kg_m3"])
    solver = Solver(h, 2, cfl * h / (wave_speed + geometry.tool_speed),
                    (geometry.initial_gap + .0005) / geometry.tool_speed, cfl)
    return {"schema_version": VERSION, "case_id": identifier, "material": card,
            "geometry": asdict(geometry), "solver": asdict(solver),
            "boundary": asdict(Boundary(True, True, 1)),
            "thermal": asdict(Thermal(True, .35, .9, mat["initial_temperature_k"], 0.)),
            "output": asdict(Output("results/cutting2d/" + identifier, 2e-6, 1000)),
            "acceptance": asdict(Acceptance()),
            "metadata": {"status": "prepared", "purpose": "Synthetic orthogonal cutting capacity; no experimental replication claim",
                         "assumptions": [
                             "Plan §8.2 dimensions, speed, angles and friction are exploratory scenario choices",
                             "Bottom one-grid-cell band fixes x/y; domain influence requires separate study",
                             "90% friction heat to workpiece, 10% to prescribed tool reservoir is an uncalibrated assumption",
                             "Zero interface conductance declares no additional temperature-driven tool exchange",
                             "Constant properties; no thermal expansion, damage, phase change or chip self-contact",
                             "Initial dt_max is a wave-speed estimate; no temporal/spatial precision has been accepted",
                             "Target travel includes initial gap plus 0.5 mm; a useful force window has not been established"],
                         "resolution_status": "unverified", "family": "basal", "baseline_id": identifier,
                         "variation": {}, "source_plan": "reports/framework_mpm_corte_2d_20260908/plan_maestro.md §8.2–8.3"}}


def capacity_cases() -> list[dict]:
    """Exactly 18 one-factor physical configurations, before any actual runs."""
    cases = []
    variations = [
        ("basal", None, None), ("speed_half", "tool_speed", 1.25), ("speed_double", "tool_speed", 5.),
        ("thickness_half", "uncut_thickness", 50e-6), ("thickness_double", "uncut_thickness", 200e-6),
        ("radius_ratio_01", "edge_radius", 10e-6), ("radius_ratio_05", "edge_radius", 50e-6),
        ("rake_5", "rake_angle_deg", 5.), ("rake_15", "rake_angle_deg", 15.)]
    for material in ("Ck45", "Ti6Al4V"):
        baseline = baseline_case(material)
        for label, key, value in variations:
            case = copy.deepcopy(baseline)
            case["case_id"] = material.lower() + "_" + label
            case["output"]["directory"] = "results/cutting2d/" + case["case_id"]
            case["metadata"]["family"] = label
            if key:
                case["geometry"][key] = value
                case["metadata"]["variation"] = {key: value}
            # Maintain the same physical travel for the speed pair; this is a
            # dependent observation duration, not a second physical parameter.
            case["solver"]["t_end"] = (case["geometry"]["initial_gap"] + .0005) / case["geometry"]["tool_speed"]
            load_case(case)
            cases.append(case)
    return cases


def explicit_case(case: dict, *, backend: str, operator: str, contact_algorithm: str) -> dict:
    """New profiles declare the complete execution route; v1 remains readable."""
    result = copy.deepcopy(case)
    result['schema_version'] = EXPLICIT_VERSION
    result['backend'] = backend
    if backend==REGISTRY_BACKEND:
        result['schema_version']=REGISTRY_VERSION
        if result['material']['schema_version']==MATERIAL_VERSION:
            result['material']['schema_version']='cutting2d-material-v2'
            result['material']['law']={'id':'finite_jc','version':1}
    result['solver'].update(operator=operator, contact_algorithm=contact_algorithm)
    return load_case(result).to_dict()


def registered_baseline(material_name: str) -> dict:
    """Prepared verification-law data; equations are selected by its card descriptor."""
    from .material_registry import synthetic_case_card
    law={'HenckyElastic':'hencky_elastic','HenckyJ2Linear':'hencky_j2_linear'}[material_name]
    case=baseline_case('SyntheticJC',case_id=law+'_basal')
    case['material']=synthetic_case_card(law)
    case['metadata']['purpose']='Verification material law; no measured alloy or accepted cutting claim'
    case['metadata']['assumptions'].append('Registered constitutive law with explicit energy contract; mixed pressure is not supported')
    return case


def write_capacity_cases(directory: str | Path, *, execution: dict | None = None) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for case in capacity_cases():
        if execution is not None:
            case = explicit_case(case, **execution)
        path = directory / (case["case_id"] + ".json")
        if path.exists():
            raise FileExistsError(f"Preserve existing case: {path}")
        path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
        entries.append({"case_id": case["case_id"], "path": path.name, "status": "prepared",
                        "canonical_sha256": load_case(case).canonical_sha256, "result": None})
    manifest = {"schema_version": "cutting2d-capacity-matrix-v1", "status": "prepared", "count": len(entries),
                "source_plan": "plan_maestro.md §8.3", "cases": entries,
                "limitations": ["No case is marked simulated or verified; precision studies remain required",
                                "SyntheticJC API demonstration is additional to the 18 configurations"]}
    (directory / "matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
