# Núcleo constitutivo del candidato Q4 directo

`finite_jc_batch.cpp` contiene la actualización numérica de
`constitutive/finite_jc/finite_jc_core.cpp` sin cambiar ley, parámetros,
búsqueda de raíces, tolerancias, cuadratura de trabajo ni guardas. El fuente
histórico permanece intacto. Las funciones Python históricas incluidas en el
archivo sirven como procedencia y para importación/exportación; el bucle
activo usa `State`, `Material` y `Options` nativos.

La versión `finite-jc-hencky-3d-stable-v2` evalúa la misma elasticidad Hencky
con `log1p` del incremento de la métrica y el flujo plástico con `expm1`
dentro de sus dominios aritméticos seguros. Fuera conserva la evaluación
original. Inicialización, actualización, lectura y validación usan la misma
aritmética; los estados de la versión anterior se rechazan. El inicializador
se expone desde este mismo módulo nativo. No se migran historias antiguas.
Los controles de precisión y de energía a deformación constante están en
`formulacion/mpm_constitutive_stable_v1` y
`formulacion/mpm_constitutive_stable_applied_v1` dentro del informe activo.

La extracción inicial y su fuente exacta se conservan en
`reports/corte_desarrollado_numerico_20260910/coste/build_v1`. El archivo
actual agrega validación nativa equivalente, buffers compartidos inmutables,
paralelismo por puntos y tangente numérico. `batch_binding.cpp.inc` corresponde
a la parte de binding; el fuente completo `finite_jc_batch.cpp` es el que
compila el loader. No regenerar desde el prototipo inicial para reemplazarlo.

`constitutive.load_native()` compila por hash con C++17, Eigen3.4.0 y
pybind11 2.13.6. Usa `-O2` sin fast-math. El hash incluye fuente, compilador,
Python y dependencias; el binario/manifiesto quedan en `.build/direct_q4`.
Reiniciar el proceso después de cambiar el código nativo: el módulo cargado
está cacheado durante la vida del proceso.

El GIL sólo se libera después de leer los buffers y antes de operar sobre
estados nativos. No se toca ningún `py::dict` dentro del bucle liberado.
Las vistas de salida tienen un propietario compartido y son de sólo lectura:
un trial posterior nunca reescribe el buffer de una respuesta anterior.
No modificar los arrays de entrada concurrentemente durante el cálculo.

Detalles, coste, limitaciones del tangente y pruebas de equivalencia están
en `reports/corte_desarrollado_numerico_20260910/coste/README.md`.
