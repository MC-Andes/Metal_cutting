"""Strict, versioned case contract. All values exposed to the solver are SI.

Bare numbers use the SI unit encoded by the schema. Explicit quantities use
{value, unit}; decimal conversion happens before floats/state initialization.
This is a common SI computation, not two native kg/g arithmetic trajectories.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

VERSION = "cutting2d-case-v1"
EXPLICIT_VERSION = "cutting2d-case-v2"
REGISTRY_VERSION = "cutting2d-case-v3"
BACKEND = "cutting2d_numpy_finite_jc_3d_v1"
REGISTRY_BACKEND = "cutting2d_numpy_material_registry_3d_v1"
OPERATORS = ("apic_b2_lumped_v1", "apic_mls_consistent_v1", "apic_mls_energy_v1", "apic_mls_energy_mixed_v1")
MATERIAL_VERSION = "cutting2d-material-v1"
UNVERIFIED_INTERFACE_ERROR = (
    "thermal.interface_conductance_w_m2_k must be zero: positive conductance is unavailable "
    "without a verified geometric contact area and interface quadrature; friction heat "
    "partition remains available"
)

# Exact decimal factors map an explicitly dimensioned input to the canonical SI unit.
UNIT_FACTORS = {
    "m": {"m": "1", "mm": ".001", "um": ".000001", "µm": ".000001"},
    "s": {"s": "1", "ms": ".001", "us": ".000001", "ns": ".000000001"},
    "m/s": {"m/s": "1", "mm/s": ".001", "m/min": "1/60"},
    "kg/m3": {"kg/m3": "1", "g/m3": ".001", "g/cm3": "1000"},
    "Pa": {"Pa": "1", "MPa": "1000000", "GPa": "1000000000", "g/(m s2)": ".001"},
    "J/(kg K)": {"J/(kg K)": "1", "J/(g K)": "1000", "m2/(s2 K)": "1"},
    "W/(m K)": {"W/(m K)": "1", "mW/(m K)": ".001"},
    "W/(m2 K)": {"W/(m2 K)": "1"},
    "K": {"K": "1"}, "deg": {"deg": "1"},
    "1/s": {"1/s": "1"}, "1": {"1": "1"},
    "J": {"J": "1", "mJ": ".001", "nJ": ".000000001"},
    "N s": {"N s": "1", "kg m/s": "1", "g m/s": ".001"},
}
MATERIAL_UNITS = {
    "density_kg_m3": "kg/m3", "young_modulus_pa": "Pa", "poisson_ratio": "1",
    "initial_temperature_k": "K", "jc_a_pa": "Pa", "jc_b_pa": "Pa",
    "jc_c": "1", "jc_m": "1", "jc_n": "1", "reference_strain_rate_s_1": "1/s",
    "reference_temperature_k": "K", "melt_temperature_k": "K",
    "specific_heat_j_kg_k": "J/(kg K)", "thermal_conductivity_w_m_k": "W/(m K)",
    "taylor_quinney": "1",
}


class ConfigurationError(ValueError):
    """Invalid input; never repaired or silently clipped."""


def _keys(value: Any, required: set[str], optional: set[str], path: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path}: expected object")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        raise ConfigurationError(f"{path}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def quantity(value: Any, unit: str, path: str = "quantity") -> float:
    """Round a dimension-checked decimal quantity once into Float64 SI."""
    factor = "1"
    if isinstance(value, dict):
        _keys(value, {"value", "unit"}, set(), path)
        if value["unit"] not in UNIT_FACTORS[unit]:
            raise ConfigurationError(f"{path}: unit {value['unit']!r} incompatible with {unit}")
        factor, value = UNIT_FACTORS[unit][value["unit"]], value["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ConfigurationError(f"{path}: expected finite number or dimensioned quantity")
    try:
        decimal = Decimal(str(value))
        if not decimal.is_finite():
            raise ConfigurationError(f"{path}: nonfinite value")
        result = float(Fraction(decimal) * Fraction(factor))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{path}: expected finite decimal") from exc
    if not decimal.is_finite() or not math.isfinite(result):
        raise ConfigurationError(f"{path}: nonfinite value")
    if decimal != 0 and result == 0:
        raise ConfigurationError(f"{path}: value underflows Float64")
    return result


def _positive(value: float, path: str, *, zero: bool = False) -> None:
    if (value < 0) if zero else (value <= 0):
        raise ConfigurationError(f"{path}: must be {'nonnegative' if zero else 'positive'}")


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ConfigurationError(f"{path}: expected explicit boolean")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{path}: expected integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Geometry:
    length: float
    height: float
    width: float
    uncut_thickness: float
    edge_radius: float
    rake_angle_deg: float
    clearance_angle_deg: float
    initial_gap: float
    tool_speed: float


@dataclass(frozen=True)
class Solver:
    h: float
    particles_per_axis: int
    dt_max: float
    t_end: float
    cfl: float
    dt_min: float = 1e-14
    max_steps: int = 1000000
    max_retries: int = 12
    operator: str = "apic_b2_lumped_v1"
    contact_algorithm: str = "domain_corners_v1"


@dataclass(frozen=True)
class Boundary:
    fix_bottom_x: bool
    fix_bottom_y: bool
    fixed_bottom_layers: int
    support_operator: str = "nodal_band_v1"


@dataclass(frozen=True)
class Thermal:
    conduction: bool
    friction_coefficient: float
    friction_heat_workpiece_fraction: float
    tool_temperature_k: float
    interface_conductance_w_m2_k: float
    conduction_operator: str = "flip_b2_v1"
    complement_rate_factor: float = 1.


@dataclass(frozen=True)
class Output:
    directory: str
    interval: float
    checkpoint_interval: int


@dataclass(frozen=True)
class Acceptance:
    mass_relative: float = 1e-12
    momentum_relative: float = 1e-6
    energy_relative: float = .05
    deformation_relative: float = .05
    penetration_over_h: float = 1e-3
    penetration_absolute_m: float = 1e-8
    energy_absolute_j: float = 1e-12
    momentum_absolute_ns: float = 1e-14
    min_plastic_strain: float = 1e-6


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    material: dict
    material_metadata: dict
    geometry: Geometry
    solver: Solver
    boundary: Boundary
    thermal: Thermal
    output: Output
    acceptance: Acceptance
    metadata: dict
    schema_version: str = VERSION
    backend: str | None = None

    def to_dict(self) -> dict:
        card = copy.deepcopy(self.material_metadata)
        card["properties"] = copy.deepcopy(self.material)
        result = {"schema_version": self.schema_version, "case_id": self.case_id, "material": card,
                **{k: asdict(getattr(self, k)) for k in (
                    "geometry", "solver", "boundary", "thermal", "output", "acceptance")},
                "metadata": copy.deepcopy(self.metadata)}
        # Preserve canonical v1 files produced before an alternative existed.
        if self.schema_version in {EXPLICIT_VERSION,REGISTRY_VERSION}:
            result["backend"] = self.backend
        if self.schema_version == VERSION and result["solver"]["operator"] == "apic_b2_lumped_v1":
            del result["solver"]["operator"]
        if self.schema_version == VERSION and result["solver"]["contact_algorithm"] == "domain_corners_v1":
            del result["solver"]["contact_algorithm"]
        # Preserve the canonical identities of pre-existing thermal cases.
        if result["boundary"]["support_operator"] == "nodal_band_v1":
            del result["boundary"]["support_operator"]
        if result["thermal"]["conduction_operator"] == "flip_b2_v1":
            del result["thermal"]["conduction_operator"]
            del result["thermal"]["complement_rate_factor"]
        return result

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_material(card: dict) -> tuple[dict, dict]:
    _keys(card, {"schema_version", "id", "law", "properties", "provenance", "validity"}, set(), "material")
    if card["schema_version"] != MATERIAL_VERSION or card["law"] != "finite_jc":
        raise ConfigurationError("material: unsupported schema version or law")
    if not isinstance(card["id"], str) or not card["id"]:
        raise ConfigurationError("material.id: nonempty string required")
    source = _keys(card["properties"], set(MATERIAL_UNITS) | {"name"}, set(), "material.properties")
    if not isinstance(source["name"], str) or not source["name"]:
        raise ConfigurationError("material.properties.name: nonempty string required")
    resolved = {k: quantity(source[k], unit, "material." + k) for k, unit in MATERIAL_UNITS.items()}
    resolved["name"] = source["name"]
    for k, v in resolved.items():
        if k != "name" and k not in {"poisson_ratio", "jc_b_pa", "jc_c", "taylor_quinney", "thermal_conductivity_w_m_k"}:
            _positive(v, "material." + k)
    for k in ("jc_b_pa", "jc_c", "thermal_conductivity_w_m_k"):
        _positive(resolved[k], "material." + k, zero=True)
    if not -1 < resolved["poisson_ratio"] < .5 or not 0 <= resolved["taylor_quinney"] <= 1:
        raise ConfigurationError("material: require -1 < nu < 0.5 and 0 <= beta <= 1")
    if not 0 < resolved["reference_temperature_k"] < resolved["melt_temperature_k"]:
        raise ConfigurationError("material: require 0 < Tref < Tm")
    if not 0 < resolved["initial_temperature_k"] < resolved["melt_temperature_k"]:
        raise ConfigurationError("material: initial temperature must be positive and below melt")
    provenance = _keys(card["provenance"], {"fields", "limitations", "constitutive_convention"}, set(), "material.provenance")
    convention_fields = {"kinematics", "stress", "rate_factor", "thermal_factor", "beta", "solid_range", "properties"}
    conventions = _keys(provenance["constitutive_convention"], convention_fields, set(), "material.constitutive_convention")
    if any(not isinstance(v, str) or not v for v in conventions.values()):
        raise ConfigurationError("material.constitutive_convention: nonempty strings required")
    if not isinstance(provenance["limitations"], list) or any(not isinstance(v, str) for v in provenance["limitations"]):
        raise ConfigurationError("material.provenance.limitations: expected list of strings")
    evidence = _keys(provenance["fields"], set(MATERIAL_UNITS), set(), "material.provenance.fields")
    for k, item in evidence.items():
        _keys(item, {"status", "source", "source_locator", "assumptions"},
              {"value", "unit", "method", "uncertainty", "limitations"}, "material.evidence." + k)
        if item["status"] not in {"published_literature", "assumed", "derived", "synthetic"}:
            raise ConfigurationError("material.evidence: unsupported provenance status")
        if item["status"] in {"published_literature", "derived"} and not item["source"]:
            raise ConfigurationError("material.evidence: published/derived field requires source")
        if item["status"] in {"assumed", "synthetic"} and not item["assumptions"]:
            raise ConfigurationError("material.evidence: assumed/synthetic field requires explicit reason")
        if not item["source_locator"] or not isinstance(item["assumptions"], list):
            raise ConfigurationError("material.evidence: source locator and assumption list required")
        if "value" in item:
            evidence_value = quantity({"value": item["value"], "unit": item.get("unit", MATERIAL_UNITS[k])}, MATERIAL_UNITS[k])
            if evidence_value != resolved[k]:
                raise ConfigurationError(f"material.evidence.{k}: provenance value differs from resolved property")
    validity = _keys(card["validity"], {"status", "temperature_k", "experimental_plastic_strain", "experimental_strain_rate_s_1"}, set(), "material.validity")
    if validity["status"] not in {"ranges_not_established", "synthetic_control", }:
        raise ConfigurationError("material.validity: unknown status")
    for k in ("temperature_k", "experimental_plastic_strain", "experimental_strain_rate_s_1"):
        pair = validity[k]
        if pair is not None:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ConfigurationError(f"material.validity.{k}: expected [lower, upper] or null")
            values = [quantity(v, "1", "material.validity." + k) for v in pair]
            if values[0] < 0 or values[0] >= values[1]:
                raise ConfigurationError(f"material.validity.{k}: invalid increasing range")
    if validity["experimental_plastic_strain"] is not None or validity["experimental_strain_rate_s_1"] is not None:
        raise ConfigurationError("material.validity: experimental ranges require a future source-checked schema version")
    if validity["temperature_k"] != [resolved["reference_temperature_k"], resolved["melt_temperature_k"]]:
        raise ConfigurationError("material.validity.temperature_k: must record [Tref,Tm], not an experimental range")
    metadata = copy.deepcopy(card)
    del metadata["properties"]
    return resolved, metadata


def load_case(source: str | Path | dict | CaseConfig) -> CaseConfig:
    if isinstance(source, CaseConfig):
        source = source.to_dict()
    if isinstance(source, (str, Path)):
        def pairs_unique(pairs):
            result = {}
            for k, value in pairs:
                if k in result:
                    raise ConfigurationError(f"Duplicate JSON key {k!r}")
                result[k] = value
            return result
        source = json.loads(Path(source).read_text(), object_pairs_hook=pairs_unique)
    source = copy.deepcopy(source)
    _keys(source, {"schema_version", "case_id", "material", "geometry", "solver", "boundary", "thermal", "output", "acceptance", "metadata"}, {"backend"}, "case")
    version = source["schema_version"]
    if version not in {VERSION, EXPLICIT_VERSION,REGISTRY_VERSION}:
        raise ConfigurationError("case: unsupported schema version")
    if version == EXPLICIT_VERSION and source.get("backend") != BACKEND:
        raise ConfigurationError(f"case.backend: v2 requires explicit {BACKEND!r}")
    if version == REGISTRY_VERSION and source.get("backend") != REGISTRY_BACKEND:
        raise ConfigurationError(f"case.backend: v3 requires explicit {REGISTRY_BACKEND!r}")
    if version == VERSION and "backend" in source:
        raise ConfigurationError("case.backend: explicit backend requires case schema v2")
    case_id = source["case_id"]
    if not isinstance(case_id, str) or not case_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in case_id):
        raise ConfigurationError("case_id: use nonempty letters, digits, dash, dot or underscore")
    if version==REGISTRY_VERSION:
        from .material_registry import validate_case_card
        material,material_metadata=validate_case_card(source['material'])
    else:material, material_metadata = validate_material(source["material"])
    g = _keys(source["geometry"], {f.name for f in fields(Geometry)}, set(), "geometry")
    g = {k: quantity(v, "deg" if k.endswith("_deg") else "m/s" if k == "tool_speed" else "m", "geometry." + k) for k, v in g.items()}
    for k in ("length", "height", "width", "uncut_thickness", "edge_radius", "tool_speed"):
        _positive(g[k], "geometry." + k)
    _positive(g["initial_gap"], "geometry.initial_gap", zero=True)
    if not g["uncut_thickness"] < g["height"]:
        raise ConfigurationError("geometry: uncut thickness must be smaller than piece height")
    if not -45 < g["rake_angle_deg"] < 60 or not 0 < g["clearance_angle_deg"] < 45 or g["rake_angle_deg"] + g["clearance_angle_deg"] >= 85:
        raise ConfigurationError("geometry: angles outside declared convex-wedge generation range")
    geometry = Geometry(**g)
    s = _keys(source["solver"], {"h", "particles_per_axis", "dt_max", "t_end", "cfl"}, {"dt_min", "max_steps", "max_retries", "operator", "contact_algorithm"}, "solver")
    if version in {EXPLICIT_VERSION,REGISTRY_VERSION} and not {"operator", "contact_algorithm"} <= s.keys():
        raise ConfigurationError("solver: v2 requires explicit operator and contact_algorithm")
    solver = Solver(h=quantity(s["h"], "m", "solver.h"),
                    particles_per_axis=_integer(s["particles_per_axis"], "solver.particles_per_axis", 1),
                    dt_max=quantity(s["dt_max"], "s", "solver.dt_max"),
                    t_end=quantity(s["t_end"], "s", "solver.t_end"),
                    cfl=quantity(s["cfl"], "1", "solver.cfl"),
                    dt_min=quantity(s.get("dt_min", 1e-14), "s", "solver.dt_min"),
                    max_steps=_integer(s.get("max_steps", 1000000), "solver.max_steps", 1),
                    max_retries=_integer(s.get("max_retries", 12), "solver.max_retries", 0),
                    operator=s.get("operator", "apic_b2_lumped_v1"),
                    contact_algorithm=s.get("contact_algorithm", "domain_corners_v1"))
    if solver.operator not in OPERATORS:
        raise ConfigurationError("solver.operator: unknown versioned operator")
    if solver.contact_algorithm not in {"domain_corners_v1", "signed_features_v1"}:
        raise ConfigurationError("solver.contact_algorithm: unknown versioned contact algorithm")
    if solver.contact_algorithm == "signed_features_v1" and solver.operator not in {"apic_mls_energy_v1", "apic_mls_energy_mixed_v1"}:
        raise ConfigurationError("solver.contact_algorithm: signed features require an energy-coordinate operator")
    for k in ("h", "dt_max", "t_end", "dt_min"):
        _positive(getattr(solver, k), "solver." + k)
    if not 0 < solver.cfl <= .5 or solver.dt_min > solver.dt_max:
        raise ConfigurationError("solver: require 0 < cfl <= 0.5 and dt_min <= dt_max")
    if solver.h > min(geometry.length, geometry.height):
        raise ConfigurationError("solver.h: exceeds the smaller piece dimension")
    b = _keys(source["boundary"], {"fix_bottom_x", "fix_bottom_y", "fixed_bottom_layers"},
              {"support_operator"}, "boundary")
    boundary = Boundary(_boolean(b["fix_bottom_x"], "boundary.fix_bottom_x"),
                        _boolean(b["fix_bottom_y"], "boundary.fix_bottom_y"),
                        _integer(b["fixed_bottom_layers"], "boundary.fixed_bottom_layers"),
                        b.get("support_operator", "nodal_band_v1"))
    if boundary.support_operator not in {"nodal_band_v1", "material_trace_v1"}:
        raise ConfigurationError("boundary.support_operator: unknown operator")
    if boundary.support_operator == "material_trace_v1":
        if solver.operator not in {"apic_mls_energy_v1", "apic_mls_energy_mixed_v1"}:
            raise ConfigurationError("material_trace_v1 requires an energy-coordinate operator")
        if boundary.fixed_bottom_layers != 0:
            raise ConfigurationError("material_trace_v1 requires fixed_bottom_layers=0; no nodal band")
        if not (boundary.fix_bottom_x or boundary.fix_bottom_y):
            raise ConfigurationError("material_trace_v1 requires at least one fixed material component")
    if boundary.fixed_bottom_layers * solver.h >= geometry.height - geometry.uncut_thickness:
        raise ConfigurationError("boundary: fixed bottom band reaches the cutting region")
    thermal_options = {"conduction_operator", "complement_rate_factor"}
    t = _keys(source["thermal"], {f.name for f in fields(Thermal)} - thermal_options, thermal_options, "thermal")
    thermal = Thermal(_boolean(t["conduction"], "thermal.conduction"),
                      quantity(t["friction_coefficient"], "1", "thermal.friction_coefficient"),
                      quantity(t["friction_heat_workpiece_fraction"], "1", "thermal.friction_heat_workpiece_fraction"),
                      quantity(t["tool_temperature_k"], "K", "thermal.tool_temperature_k"),
                      quantity(t["interface_conductance_w_m2_k"], "W/(m2 K)", "thermal.interface_conductance_w_m2_k"),
                      t.get("conduction_operator", "flip_b2_v1"),
                      quantity(t.get("complement_rate_factor", 1.), "1", "thermal.complement_rate_factor"))
    if thermal.conduction_operator not in {"flip_b2_v1", "projected_galerkin_v1"}:
        raise ConfigurationError("thermal.conduction_operator: unknown operator")
    _positive(thermal.complement_rate_factor, "thermal.complement_rate_factor")
    if thermal.conduction_operator == "flip_b2_v1" and thermal.complement_rate_factor != 1.:
        raise ConfigurationError("thermal.complement_rate_factor only applies to projected_galerkin_v1")
    if thermal.conduction_operator == "projected_galerkin_v1" and "complement_rate_factor" not in t:
        raise ConfigurationError("projected_galerkin_v1 requires explicit thermal.complement_rate_factor")
    _positive(thermal.friction_coefficient, "thermal.friction_coefficient", zero=True)
    _positive(thermal.interface_conductance_w_m2_k, "thermal.interface_conductance_w_m2_k", zero=True)
    if thermal.interface_conductance_w_m2_k > 0:
        raise ConfigurationError(UNVERIFIED_INTERFACE_ERROR)
    _positive(thermal.tool_temperature_k, "thermal.tool_temperature_k")
    if not 0 <= thermal.friction_heat_workpiece_fraction <= 1:
        raise ConfigurationError("thermal: workpiece heat fraction must be within [0,1]")
    if thermal.conduction and material["thermal_conductivity_w_m_k"] <= 0:
        raise ConfigurationError("thermal: conduction requires positive conductivity")
    o = _keys(source["output"], {f.name for f in fields(Output)}, set(), "output")
    if not isinstance(o["directory"], str) or not o["directory"].strip():
        raise ConfigurationError("output.directory: nonempty path required")
    output = Output(o["directory"], quantity(o["interval"], "s", "output.interval"),
                    _integer(o["checkpoint_interval"], "output.checkpoint_interval"))
    _positive(output.interval, "output.interval")
    a = _keys(source["acceptance"], {f.name for f in fields(Acceptance)}, set(), "acceptance")
    acceptance_units = {"penetration_absolute_m": "m", "energy_absolute_j": "J", "momentum_absolute_ns": "N s"}
    acceptance = Acceptance(**{k: quantity(v, acceptance_units.get(k, "1"), "acceptance." + k) for k, v in a.items()})
    for k, v in asdict(acceptance).items():
        _positive(v, "acceptance." + k)
    metadata = _keys(source["metadata"], {"status", "purpose", "assumptions", "resolution_status"}, {"family", "baseline_id", "variation", "source_plan"}, "metadata")
    if metadata["status"] not in {"prepared", "diagnostic"} or metadata["resolution_status"] != "unverified":
        raise ConfigurationError("metadata: input may only claim prepared/diagnostic and unverified resolution")
    if not isinstance(metadata["purpose"], str) or not isinstance(metadata["assumptions"], list) or not metadata["assumptions"]:
        raise ConfigurationError("metadata: purpose and explicit nonempty assumptions required")
    if any(not isinstance(v, str) or not v for v in metadata["assumptions"]):
        raise ConfigurationError("metadata.assumptions: nonempty strings required")
    if "variation" in metadata:
        _keys(metadata["variation"], set(), {f.name for f in fields(Geometry)}, "metadata.variation")
        for key, value in metadata["variation"].items():
            quantity(value, "deg" if key.endswith("_deg") else "m/s" if key == "tool_speed" else "m", "metadata.variation." + key)
    config = CaseConfig(case_id, material, material_metadata, geometry, solver, boundary, thermal, output, acceptance, metadata, version, source.get("backend"))
    if version==REGISTRY_VERSION and solver.operator=='apic_mls_energy_mixed_v1':
        from .material_registry import case_capabilities
        if not case_capabilities(config)['production_mixed_operator']:
            raise ConfigurationError('material/operator capabilities: registered law has no verified mixed-pressure adapter')
    # Geometric feasibility is independent of mesh and checked at the same contract boundary.
    from .cases import make_tool
    make_tool(config)
    return config
