# Revisión del flujo común — 2026-10-16

Esta revisión comprueba ejecución y trazabilidad con archivos locales. No es una
validación científica del método ni una aceptación de los eventos como fases.

## Verificaciones sintéticas

`tests/test_common_calorimetry.py` y `tests/test_common_calorimetry_app.py` cubren:

- Intersección conocida con línea base inclinada, signos positivo/negativo,
  calentamiento/enfriamiento y pasos de 0.1, 0.5 y 2 s.
- Tres eventos manuales con amplitudes distintas y cuatro propuestas automáticas;
  no existe un límite de dos ni sustitución de una selección por el mayor pico.
- Señal manual débil que conserva resultado geométrico aunque no cumpla SNR.
- Ruido sin desviación persistente, con fallo explícito de selección automática
  y posibilidad de introducir regiones manuales.
- Sinusoide con indicador de periodicidad y muestreo irregular no evaluado como
  oscilación. No prueba que una oscilación sea instrumental.
- Isoterma, variables ausentes, unidades W/g conservadas y mW sin conversión.
- Persistencia limitada a la región del candidato, sin unir máscaras disjuntas.
- Fallos por huecos, NaN, reloj repetido/invertido y filas de origen omitidas.
- Interpolación solo entre muestras adyacentes, sin extrapolación.
- Lectores Calisto/TRIOS con fixtures sintéticos y parser TRIOS simulado en la
  frontera del adaptador; esa prueba no constituye revisión de un `.xls` real.
- Bundle con fuentes archivadas, hashes, revisión, rechazo de sobrescritura,
  gráficos sin llamadas al cálculo y comparación con control de unidades.
- Entrada CLI, revisión manual de etapas, editor y tres acciones de Streamlit.

La suite histórica se ejecutó junto con las pruebas nuevas: 84 pruebas pasaron
en la verificación completa (incluidas 43 del flujo común). Las pruebas
históricas siguen comprobando los métodos antiguos, no equivalencia numérica
con el método común.

## LabVIEW/Calvet: `26_08_17`

Se cargaron los archivos locales completos, sin recortar al programa PDF:

| Fuente | SHA256 |
|---|---|
| `data/raw/lvm/26_08_17.lvm` | `d025115d5367f58c03f16f201c70ae0167c40dea0191e824421fc12a618802b2` |
| `data/raw/pdf/26_08_17.pdf` | `e1230d1ae10c2c3ff31cdf1d6b6d580e2d60c41a77670af4dbad3c0b3fc843fd` |

294587 filas. Se seleccionó explícitamente el reloj por índice a 0.5 s por
muestra. El reloj registrado queda guardado, con sus duplicados y diferencia
respecto al índice. La identificación de columnas LVM conserva la heurística
existente y requiere contraste con la exportación instrumental.

Se usó `data/event_selections/26_08_17_review.json`; ambas regiones siguen
`pending_review`, sin hipótesis LL/SL:

| Región exploratoria | Línea base (s) | Flanco (s) | Intersección (s) | T de muestra (°C) | P (bar) |
|---|---|---|---:|---:|---:|
| Débil | 118800–122400 | 124020–125280 | 122886.952 | 26.261 | 310.010 |
| Grande | 140400–141840 | 142380–142560 | 142378.599 | 10.400 | 206.500 |

Los valores 25.43 y 10.41 °C suministrados previamente no se utilizaron como
restricciones del ajuste. La diferencia en la región débil muestra que marcar
un tiempo por temperatura suministrada y estimar un onset por intersección son
operaciones distintas. Su flanco tiene RMSE ≈0.02354 mW y R² ≈0.8684; no debe
aceptarse solo porque su SNR operativo sea alto.

En la región grande, otra elección exploratoria de línea base 139680–141480 s
y flanco 142488–142740 s dio ≈10.511 °C y 208.3 bar. Es una estimación alternativa
del **mismo evento**, no otra transición. La sensibilidad real a la selección de
ventanas requiere revisión; el diagnóstico automático de quitar un punto de
cada extremo no cubre este cambio de regiones.

Resultados y gráficos locales, ignorados por Git:

- `out/common_review_20261016/calvet_signals/results.json`
- `out/common_review_20261016/calvet_analysis/results.json`
- `out/common_review_20261016/large_region.png`
- `out/common_review_20261016/weak_region.png`
- `out/common_review_20261016/ui_review/results.json` (revisión mediante la interfaz)

La ejecución de propuestas automáticas en la etapa 4, con separación 600 s y
semiancho de flanco 30 s, generó **37 candidatos** en
`out/common_review_20261016/calvet_candidates/`. Esto muestra que todavía
propone muchas regiones que requieren filtrado y revisión; no son 37 eventos
físicos confirmados. No se sustituyeron las ventanas manuales por esas propuestas.

Los dos ajustes se inspeccionaron gráficamente con HF original, regiones,
rectas e intersección. La revisión mediante Streamlit también guardó ambas
selecciones sin excepciones. Esto comprueba funcionamiento y trazabilidad;
no establece exactitud de onset, identidad de fases ni estabilidad de equilibrio.

## Calisto: `23_05_12`

Se cargaron `data/raw/lvm/23_05_12.txt` y `data/raw/pdf/23_05_12.pdf` con el lector
específico: 577211 filas. La temperatura disponible es de **horno**, no de
muestra, y no hay presión. El lector informa 482 valores de tiempo duplicados
y 57388 intervalos no positivos; se conserva el orden original.

El intento de ajustar la ventana de tiempo 0–100 s sin selección de filas se
guardó como `not_determined`, motivo `Non-increasing acquisition time`. No se
ordenaron los datos ni se inventó correspondencia de ciclo con el PDF.
Una selección `row_range` permite revisar un tramo contiguo monotónico; la
identificación experimental del ciclo sigue pendiente.

También se revisó una corrección manual dentro de las filas canónicas inclusivas
62663–76007. Para ventana 33000–34500 s y línea base 33100–33600 s, un flanco
posterior al salto (33950–33960 s) falló por intersección fuera de rango. Al
seleccionar el ascenso inicial 33943.5–33947.7 s, el cálculo dio 33943.847 s y
**15.314 °C de horno**. Temperatura de muestra y presión siguieron no disponibles.
La selección permanece pendiente de revisión, sin asignación de fase ni
validación de correspondencia con el PDF.

Bundles locales:

- `out/common_review_20261016/calisto_signals/results.json`
- `out/common_review_20261016/calisto_clock_review/results.json`
- `out/common_review_20261016/calisto_manual_review/results.json` (fallo de geometría)
- `out/common_review_20261016/calisto_refined_review/results.json` (región corregida)
- `out/common_review_20261016/calisto_refined_event.png`

## TRIOS y tareas pendientes

No se encontró un `.xls` TRIOS experimental en `data/` de este checkout. Se
verificaron el adaptador y la reconstrucción opcional con fixtures sintéticos;
queda pendiente el recorrido completo con un workbook experimental real.

También siguen pendientes la validación experimental de umbrales, convenciones
endo/exo, sincronización instrumental, incertidumbre calibrada, separación de
solapamientos y repetibilidad con ensayos/ciclos identificados. Los indicadores
no convierten automáticamente ruido débil en descarte ni periodicidad en
oscilación instrumental confirmada.
