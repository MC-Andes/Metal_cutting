#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Uso: $0 <quick|reference>" >&2
  exit 1
fi

CASE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE_DIR="${REPO_ROOT}/cases/cutting2d/${CASE_NAME}"

if [[ ! -d "${CASE_DIR}" ]]; then
  echo "No existe el caso ${CASE_NAME}. Ejecuta scripts/generate_repro_case.py primero." >&2
  exit 1
fi

if [[ -z "${KRATOS_ROOT:-}" ]]; then
  echo "Define KRATOS_ROOT apuntando a tu compilación de Kratos." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${KRATOS_ROOT}:${PYTHONPATH:-}"

pushd "${CASE_DIR}" >/dev/null
"${PYTHON_BIN}" "${KRATOS_ROOT}/kratos/python_scripts/MainKratos.py" > run.log 2>&1
popd >/dev/null

echo "Caso ${CASE_NAME} ejecutado. Revisa ${CASE_DIR}/run.log"
