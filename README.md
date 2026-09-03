# Metal_cutting

Caso reproducible para corte ortogonal 2D en Kratos (PFEM Solid).

## Archivos clave

- `cases/cutting2d/case_parameters.json`: parámetros centralizados.
- `scripts/audit_legacy_cutting_example.py`: auditoría automática del caso legacy `cutting_test_rigid_2D.gid`.
- `scripts/generate_repro_case.py`: genera `quick` y `reference`.
- `scripts/run_case.sh`: ejecuta un caso con `KRATOS_ROOT`.
- `scripts/postprocess_cutting_metrics.py`: calcula métricas de fuerzas, energía y viruta.
- `reports/informe_tecnico_es.md`: informe técnico en español.

## Flujo rápido

```bash
python3 scripts/audit_legacy_cutting_example.py
python3 scripts/generate_repro_case.py
KRATOS_ROOT=/ruta/a/kratos scripts/run_case.sh quick
```

Postproceso:

```bash
python3 scripts/postprocess_cutting_metrics.py \
  --forces /ruta/forces.csv \
  --energies /ruta/energies.csv \
  --chip /ruta/chip.csv \
  --transient-cutoff-time 1.2e-5 \
  --output /ruta/summary.json
```
