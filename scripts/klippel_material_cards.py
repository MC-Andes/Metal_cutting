"""Explicitly sourced exploratory JC cards, separate from the 2024 tensile fits.

``select_card(material)`` returns a fresh JSON-serializable mapping. ``material``
uses the cutting generator's SI names; ``variables`` is ready for Kratos except
for case-specific THICKNESS and MATERIAL_POINTS_PER_ELEMENT. No calibration or
experimental validation is implied. The scalar oracle is not an MPM solver.
"""
from __future__ import annotations

import copy
import math

CARD_VERSION = "klippel-exploratory-material-cards-v1"
JASPERS = "https://pure.tue.nl/ws/portalfiles/portal/1434816/9901553.pdf"
KLIPPEL2024 = "https://doi.org/10.1007/s00170-024-14597-2"
KLIPPEL2026 = "https://doi.org/10.1016/j.cirpj.2025.11.012"
KLIPPEL2026_FULLTEXT = (
    "https://www.researchgate.net/publication/400322853_"
    "Inverse_identification_of_Johnson-Cook_flow_stress_parameters_for_Ti6Al4V"
)
KRATOS_SOURCE = (
    "https://raw.githubusercontent.com/KratosMultiphysics/Kratos/"
    "96bad120e292818e24589cb5edc5a1237c89c550/applications/MPMApplication/"
    "custom_constitutive/johnson_cook_thermal_plastic_3D_law.cpp"
)

_VARIABLE_NAMES = {
    "density_kg_m3": "DENSITY",
    "young_modulus_pa": "YOUNG_MODULUS",
    "poisson_ratio": "POISSON_RATIO",
    "initial_temperature_k": "TEMPERATURE",
    "jc_a_pa": "JC_PARAMETER_A", "jc_b_pa": "JC_PARAMETER_B",
    "jc_c": "JC_PARAMETER_C", "jc_m": "JC_PARAMETER_m", "jc_n": "JC_PARAMETER_n",
    "reference_strain_rate_s_1": "REFERENCE_STRAIN_RATE",
    "reference_temperature_k": "REFERENCE_TEMPERATURE",
    # MELD is the spelling registered by Kratos, not a typo in this interface.
    "melt_temperature_k": "MELD_TEMPERATURE",
    "specific_heat_j_kg_k": "SPECIFIC_HEAT",
    "taylor_quinney": "TAYLOR_QUINNEY_COEFFICIENT",
}
_UNITS = {
    "density_kg_m3": "kg/m3", "young_modulus_pa": "Pa", "poisson_ratio": "1",
    "initial_temperature_k": "K", "jc_a_pa": "Pa", "jc_b_pa": "Pa",
    "jc_c": "1", "jc_m": "1", "jc_n": "1", "reference_strain_rate_s_1": "1/s",
    "reference_temperature_k": "K", "melt_temperature_k": "K",
    "specific_heat_j_kg_k": "J/(kg K)", "taylor_quinney": "1",
    "thermal_conductivity_w_m_k": "W/(m K)",
}

_MATERIALS = {
    "Ck45": {
        "name": "Ck45", "density_kg_m3": 7870.0, "young_modulus_pa": 205e9,
        "poisson_ratio": 0.29, "initial_temperature_k": 293.15,
        "jc_a_pa": 553.1e6, "jc_b_pa": 600.8e6, "jc_c": 0.0134,
        "jc_n": 0.234, "jc_m": 1.0, "reference_strain_rate_s_1": 1.0,
        "reference_temperature_k": 293.15, "melt_temperature_k": 1733.0,
        "specific_heat_j_kg_k": 486.0, "thermal_conductivity_w_m_k": 50.7,
        "taylor_quinney": 0.9,
    },
    "Ti6Al4V": {
        "name": "Ti6Al4V", "density_kg_m3": 4430.0, "young_modulus_pa": 110e9,
        "poisson_ratio": 0.35, "initial_temperature_k": 300.0,
        "jc_a_pa": 852.1e6, "jc_b_pa": 338.9e6, "jc_c": 0.02754,
        "jc_n": 0.148, "jc_m": 0.5961, "reference_strain_rate_s_1": 1.0,
        "reference_temperature_k": 300.0, "melt_temperature_k": 1836.0,
        "specific_heat_j_kg_k": 526.0, "thermal_conductivity_w_m_k": 6.8,
        "taylor_quinney": 0.9,
    },
}


def _field_evidence(material: str, key: str, value: float) -> dict:
    status, assumptions = "published_literature", []
    if material == "Ti6Al4V":
        source = KLIPPEL2026
        locator = "Table 13" if key.startswith("jc_") else "Table 5"
        method = "Author-deposited full text checked; PDF visual check unavailable"
        if key == "initial_temperature_k":
            status, locator = "assumed", "Table 5 reference temperature"
            assumptions = ["Initialize at the constitutive reference temperature; not measured ambient temperature"]
        if key == "jc_n":
            assumptions = ["Use Table 13 value 0.148; Table 7 rank 1 gives 0.1483; preserve rounding discrepancy"]
    elif key.startswith("jc_") or key in {"melt_temperature_k", "reference_strain_rate_s_1"}:
        source = JASPERS
        locator = ("p. 92 Eq. 5.7 (PDF p. 110)" if key == "reference_strain_rate_s_1"
                   else "p. 94 Table 5.3 (PDF p. 112)")
        method = "Direct visual transcription of original thesis"
        assumptions = ["Normalized AISI 1045 is a different-lot proxy for Ck45"]
    elif key in {"reference_temperature_k", "initial_temperature_k"}:
        source, locator = JASPERS, "pp. 89, 92: room temperature and Fig. 5.9 at 20 degC"
        status, method = "derived", "20 degC + 273.15 = 293.15 K"
        assumptions = ["Room temperature is represented by 20 degC; not measured Klippel ambient"]
    elif key == "taylor_quinney":
        source, locator = None, "Exploratory model assumption"
        status, method = "assumed", "Set 90 percent plastic-work heat conversion, to be sensitivity-tested"
        assumptions = ["Not a Ck45 batch measurement or part of Jaspers fitted card; Jaspers p. 93 assumed all deformation energy becomes heat"]
    else:
        source, locator = KLIPPEL2024, "Table 10 and section 2.4.5"
        method = "Published typical literature constants, not measurements of the batch"
    return {
        "value": value, "unit": _UNITS[key], "status": status, "source": source,
        "source_locator": locator, "method": method, "assumptions": assumptions,
        "uncertainty": None,
        "limitations": ["No identified parameter covariance or uncertainty; no MPM calibration"],
    }


def select_card(material: str) -> dict:
    """Return a complete sourced candidate; reject unknown materials explicitly."""
    names = {"ck45": "Ck45", "aisi1045": "Ck45", "ti6al4v": "Ti6Al4V"}
    normalized = str(material).lower().replace("-", "").replace(" ", "")
    if normalized not in names:
        raise ValueError(f"Unsupported material {material!r}; expected Ck45 or Ti6Al4V")
    material = names[normalized]
    m = copy.deepcopy(_MATERIALS[material])
    ck45 = material == "Ck45"
    card_id = ("ck45_literature_jaspers1999_normalized_aisi1045_proxy"
               if ck45 else "ti6al4v_literature_inverse_identified_sph_2026")
    limits = [
        "Exploratory candidate: successful execution does not establish experimental validation",
        "Isotropic separable Johnson-Cook: no calibrated damage/fracture or length scale; Kratos has an internal yield/virgin-yield cutoff at 1e-3",
        "Thermal conductivity is metadata; this card does not implement conduction",
        "Kratos clamps strain-rate hardening at 1 below reference rate; original logarithm is not extrapolated downward",
        "Kratos beta=0 disables thermal softening as well as adiabatic heat generation",
        "Kratos MP_TEMPERATURE setter forbids updates after plastic strain; friction-heat deposition requires a constitutive implementation change",
    ]
    if ck45:
        limits += [
            "Different normalized AISI1045 lot: not the measured 2024 Ck45 tensile fit",
            "Primary thesis documents dynamic strain-aging and pronounced JC discrepancies around 500 degC",
            "Tm=1733 K from original Table5.3; do not substitute Klippel2024 typical1773 K or Burns2013 value1490 degC",
            "Elastic and thermophysical constants use Klippel2024 typical literature values; beta=.9 is an explicit additional assumption",
        ]
    else:
        limits += [
            "Identified in SPH using cutting condition V0060; transfer to MPM/contact/treatment requires validation",
            "V0060 is not a held-out validation test and its 2024 catalogue status is short",
            "Identification includes workpiece conduction but initially omits frictional heating and tool conduction",
            "Table13 n=.148 differs from Table7 n=.1483; use the summarized card without silent extra precision",
        ]
    return {
        "version": CARD_VERSION, "card_id": card_id,
        "status": "exploratory_not_calibrated_in_mpm", "material": m,
        "constitutive_law": "JohnsonCookThermalPlastic2DPlaneStrainLaw",
        "variables": {variable: m[key] for key, variable in _VARIABLE_NAMES.items()},
        "provenance": {
            "fields": {key: _field_evidence(material, key, value) for key, value in m.items() if key != "name"},
            "limitations": limits, "kratos_source": KRATOS_SOURCE,
            "source_evidence_report": "reports/klippel_material_cards_20260908/informe.md",
            "source_fulltext": JASPERS if ck45 else KLIPPEL2026_FULLTEXT,
        },
        "contact_candidate": {
            "coulomb_friction": 0.35,
            "status": "assumed_not_calibrated" if ck45 else "published_model_choice_not_measurement",
            "source": None if ck45 else KLIPPEL2026,
            "source_locator": "Exploratory pilot assumption" if ck45 else "Table5",
            "limitations": ["Not the apparent force ratio Ff/Fc; requires independent sensitivity"],
        },
    }


def flow_stress_pa(card: dict, plastic_strain: float, plastic_strain_rate_s_1: float,
                   temperature_k: float) -> float:
    """Scalar oracle for the documented Kratos yield surface, not stress integration.

    This evaluates the mathematical surface independently of Kratos. Tests of
    this helper cannot certify return mapping, objectivity or MPM transfer.
    """
    ep, rate, temp = map(float, (plastic_strain, plastic_strain_rate_s_1, temperature_k))
    if not all(math.isfinite(x) for x in (ep, rate, temp)) or ep < 0 or rate < 0 or temp <= 0:
        raise ValueError("Strain/rate must be finite and nonnegative; temperature positive")
    m = card["material"]
    hardening = m["jc_a_pa"] + m["jc_b_pa"] * ep ** m["jc_n"]
    rate_factor = 1 + m["jc_c"] * math.log(max(1.0, rate / m["reference_strain_rate_s_1"]))
    theta = min(1.0, max(0.0, (temp - m["reference_temperature_k"]) /
                         (m["melt_temperature_k"] - m["reference_temperature_k"])))
    thermal = 1.0 if m["taylor_quinney"] == 0 else 1 - theta ** m["jc_m"]
    return hardening * rate_factor * thermal


def adiabatic_temperature_increment_k(card: dict, plastic_work_density_j_m3: float) -> float:
    """Constant-cp heat balance beta*Wplastic/(rho*cp), no conduction/friction."""
    work = float(plastic_work_density_j_m3)
    if not math.isfinite(work) or work < 0:
        raise ValueError("Plastic work density must be finite and nonnegative")
    m = card["material"]
    return m["taylor_quinney"] * work / (m["density_kg_m3"] * m["specific_heat_j_kg_k"])
