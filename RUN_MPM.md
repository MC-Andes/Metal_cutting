# Ejecutar el MPM actual

Requiere Python 3.12, un compilador C++17 (`c++`) y las cabeceras de desarrollo de Python. Ejecutar desde la raíz del repositorio. No requiere Kratos para esta ruta MPM. Eigen 3.4.0 está incluido con su manifiesto y licencias dentro del archivo de distribución.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mpm.txt

MPM_LAUNCH=reports/corte_desarrollado_numerico_20260910/formulacion/mpm_runtime_frozen_v4/launch.py
.venv/bin/python -I "$MPM_LAUNCH" prepare runs/runtime_mpm_v7 --backend certified_range_robust --case cases/cutting2d/mpm_ck45_current.json
.venv/bin/python -I "$MPM_LAUNCH" check runs/runtime_mpm_v7 --grid-h 25e-6 --material-h 6.25e-6 --dt 100e-9 --startup-dt 25e-12 --end 1.2e-3 --wall 43200 --threads 8
.venv/bin/python -I "$MPM_LAUNCH" run runs/runtime_mpm_v7 --destination runs/corte_mpm_v7 --grid-h 25e-6 --material-h 6.25e-6 --dt 100e-9 --startup-dt 25e-12 --end 1.2e-3 --wall 43200 --threads 8
```

`prepare` compila las extensiones y congela fuentes, caso y entorno local. Debe utilizar una carpeta nueva. `check` verifica la instalación sin avanzar la simulación. `run` inicia desde t=0; también exige un destino nuevo. El modelo usa MPM con dominios Q4 compartidos, inicio explícito hasta 5 ns y luego integración implícita BE adaptativa. Los argumentos del lanzador fijan malla, integración y horizonte; el JSON contiene la física y los controles del caso.

El objetivo es 500 µm de recorrido; esta versión todavía no acredita el corte completo. Las guardas pueden reducir mucho el paso temporal y aumentar el coste de cálculo. `progress.json` y `timing_live.jsonl` muestran el avance; los registros completos y el checkpoint se cierran al terminar o alcanzar el presupuesto operativo.

Para continuar un cierre por presupuesto, conservar el mismo runtime, entorno y checkpoint. No mover el runtime congelado: sus rutas e identidad pertenecen a esa instalación. Usar una carpeta nueva para cada continuación:

```sh
MPM_RESUME=reports/corte_desarrollado_numerico_20260910/formulacion/mpm_reinicio_frozen_v3/resume.py
.venv/bin/python -I "$MPM_RESUME" runs/runtime_mpm_v7 runs/corte_mpm_v7/checkpoint.json --end 1.2e-3 --wall 43200 --check-output runs/reinicio_control_01.json
.venv/bin/python -I "$MPM_RESUME" runs/runtime_mpm_v7 runs/corte_mpm_v7/checkpoint.json --end 1.2e-3 --wall 43200 --destination runs/corte_mpm_v7_continuacion_01
```

Los dos lanzadores conservan sus rutas históricas dentro de `reports/` porque el código calcula la raíz desde su ubicación. El repositorio incluye sólo esos archivos ejecutables de esa carpeta, no el informe, las corridas ni sus archivos congelados. Una instalación nueva genera su propia identidad y no puede continuar silenciosamente checkpoints de otro entorno.
