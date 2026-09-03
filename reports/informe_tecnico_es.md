# Modelo 2D reproducible de corte ortogonal en Kratos (PFEM Solid)

## Diagnóstico inicial del caso legacy `cutting_test_rigid_2D.gid`

Se añadió una auditoría reproducible (`scripts/audit_legacy_cutting_example.py`) fijada al commit de Kratos `89db3d29a3fa0c4d45b31af1e05988b48f8ff12a`.

Resultados principales (ver `reports/legacy_cutting_audit.json`):

- Caso en formato GiD legacy (`.prb/.mat/.cnd`), no directamente reproducible solo con CLI moderna.
- Configuración detectada: `QuasiStaticSolver` con integración implícita.
- Contacto/fricción legacy no alineado con los objetivos del problema (`FindContacts=False`, `Friction_Active=False`).
- Requiere modernización a archivos JSON versionables y parametrizados.

## Implementación reproducible añadida en este repositorio

Se implementó una estructura reproducible en `cases/cutting2d` con parámetros centralizados:

- `cases/cutting2d/case_parameters.json`: geometría, herramienta, cinemática, contacto, material, remallado y salida.
- `scripts/generate_repro_case.py`: genera automáticamente los casos `quick` y `reference` con:
  - `ProjectParameters.json`
  - `Materials.json`
  - `CaseMetadata.json`
- `scripts/run_case.sh`: ejecución por CLI (`quick|reference`) usando `KRATOS_ROOT`.
- `scripts/postprocess_cutting_metrics.py`: postproceso automatizado de fuerzas, energías y métricas de viruta.

## Formulación seleccionada (mecánica)

- Dominio 2D en deformación plana.
- Análisis cuasiestático implícito como punto de partida.
- Contacto por penalización con fricción de Coulomb parametrizada.
- Material elastoplástico J2 con endurecimiento (ley `HyperElasticPlasticJ2PlaneStrain2DLaw`).
- PFEM/remallado activado y parametrizado para estudiar separación sin línea de partición prefijada.

## Casos incluidos

- **quick**: caso corto para prueba rápida de pipeline.
- **reference**: caso base de mayor duración para extraer fuerzas/energías y geometría de viruta.

Ambos se generan desde los mismos parámetros centrales, evitando valores ocultos en código.

## Postproceso y métricas

`postprocess_cutting_metrics.py` calcula automáticamente:

- Fuerza de corte media (sin transitorio).
- Fuerza de empuje media.
- Fuerza resultante y fricción aparente.
- Espesor de viruta y ratio de compresión.
- Energías final interna/cinética/trabajo externo/disipaciones.

## Limitaciones actuales y siguientes pasos

- Este repositorio no contiene el árbol completo de Kratos ni resultados numéricos ejecutados dentro de este entorno.
- La infraestructura reproducible queda preparada para ejecutar y completar:
  1. corrida de indentación/deslizamiento sin separación,
  2. activación/calibración de separación,
  3. estudios de convergencia,
  4. validación cuantitativa frente a datos publicados.

