#!/usr/bin/env python3
"""Audit Kratos legacy cutting_test_rigid_2D.gid files and emit a concise diagnosis."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "legacy_cutting_audit.json"

BASE_URL = (
    "https://raw.githubusercontent.com/KratosMultiphysics/Kratos/"
    "89db3d29a3fa0c4d45b31af1e05988b48f8ff12a/"
    "applications/PfemSolidMechanicsApplication/test_examples/cutting_test_rigid_2D.gid"
)


def _fetch_text(filename: str) -> str:
    with urllib.request.urlopen(f"{BASE_URL}/{filename}", timeout=30) as response:  # nosec B310
        return response.read().decode("utf-8")


def _match(text: str, key: str) -> str | None:
    expr = rf"QUESTION:\s*{re.escape(key)}[^\n]*\nVALUE:\s*([^\n]+)"
    m = re.search(expr, text)
    return m.group(1).strip() if m else None


def main() -> None:
    prb = _fetch_text("cutting_test_rigid_2D.prb")
    mat = _fetch_text("cutting_test_rigid_2D.mat")

    constitutive_law = _match(mat, "CONSTITUTIVE_LAW_NAME")
    solver_type = _match(prb, "Solver_Type")
    integration = _match(prb, "Time_Integration_Method")
    find_contacts = _match(prb, "FindContacts")
    friction_active = _match(prb, "Friction_Active")
    contact_method = _match(prb, "Contact_Method")

    diagnosis = {
        "source": {
            "kratos_commit": "89db3d29a3fa0c4d45b31af1e05988b48f8ff12a",
            "example_path": "applications/PfemSolidMechanicsApplication/test_examples/cutting_test_rigid_2D.gid"
        },
        "detected_configuration": {
            "solver_type": solver_type,
            "time_integration": integration,
            "contact_search_enabled": find_contacts,
            "friction_active": friction_active,
            "contact_method": contact_method,
            "constitutive_law_name_first_match": constitutive_law
        },
        "required_applications": [
            "PfemSolidMechanicsApplication",
            "SolidMechanicsApplication",
            "ContactMechanicsApplication",
            "ConstitutiveModelsApplication"
        ],
        "audit_findings": [
            "El caso está en formato GiD legacy (.prb/.mat/.cnd) y requiere modernización a JSON reproducible.",
            "La configuración detectada usa QuasiStaticSolver implícito con paso temporal muy pequeño.",
            "FindContacts=False y Friction_Active=False en el archivo legacy, lo que no satisface el objetivo físico de fricción herramienta-material.",
            "El nombre de ley constitutiva detectado incluye entradas antiguas/ambiguas en el catálogo legacy y requiere validación en Kratos actual.",
            "Se requiere auditar convergencia energética para justificar régimen cuasiestático en el caso modernizado."
        ],
        "limitations_in_this_repository": [
            "Este repositorio no incluye el código fuente completo de Kratos ni el ejemplo legacy localmente.",
            "La auditoría se realiza contra el commit fijado de Kratos por descarga reproducible."
        ]
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
