# Metal_cutting

Modelo termomecánico de corte ortogonal mediante MPM con dominios materiales
Q4 compartidos. La ruta actual usa código Python/C++ propio, plasticidad finita
Johnson–Cook, contacto con fricción y conducción térmica. Combina un arranque
explícito con integración implícita adaptativa y **no requiere Kratos**.

## Ejecutar la ruta MPM actual

La [guía RUN_MPM.md](RUN_MPM.md) explica requisitos, preparación, comprobación,
ejecución y reinicio. Requiere Python 3.12, compilador C++17 y las dependencias
fijadas en [requirements-mpm.txt](requirements-mpm.txt). El caso de desarrollo
[Ck45](cases/cutting2d/mpm_ck45_current.json) usa una velocidad de 25 m/min.

**Estado:** desarrollo y verificación numérica. El recorrido objetivo de 500 µm,
la formación de una viruta desarrollada y la validación experimental siguen
pendientes. Terminar una ejecución no implica que sus balances estén aceptados.
La guía incluye un arranque mínimo para comprobar la instalación.

## Preparación histórica PFEM/Kratos

Los archivos y comandos siguientes pertenecen a la preparación histórica de un
caso de corte ortogonal 2D en Kratos PFEM Solid. Se conservan como antecedente y
no son el procedimiento de ejecución del MPM actual.

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
