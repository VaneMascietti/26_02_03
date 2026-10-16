# Flujo común de calorimetría

Se usa `cli.py explore|analyze|plot` o `streamlit run app.py`, opción
**Common calorimetry**. El flujo común alimenta un único núcleo de onset desde
los tres lectores. Los comandos antiguos siguen disponibles con sus métodos
históricos; no son resultados equivalentes ni se migran silenciosamente.

## Cargar → seleccionar → revisar → guardar → graficar

1. **Cargar:** elegir formato, archivo y PDF cuando corresponda. Se guarda un
   bundle de señales; no se calcula ningún onset. Examinar señales, unidades,
   reloj, variables disponibles y etapas del PDF/hoja.
2. **Seleccionar:** elegir etapa o editar límites manuales. Añadir cualquier
   cantidad de filas en el editor de eventos; cada fila define una ventana en
   segundos que incluye línea base previa y flanco inicial. Editar o borrar
   filas para corregir selecciones. La detección automática solo añade propuestas.
3. **Revisar:** seleccionar explícitamente línea base y flanco, o dejarlos
   vacíos para intentar selección automática. Revisar los gráficos individuales,
   residuos, persistencia, estabilidad y geometría. Un fallo conserva su motivo.
4. **Guardar:** cada cálculo se guarda en una revisión nueva, incluyendo la
   selección y revisión manual. En la interfaz, **Calculate, save and review
   fits** guarda una revisión antes de mostrar los ajustes. Para corregirla,
   abrir esa revisión, editar y guardar en otro directorio.
5. **Graficar:** **Plot saved results** modifica presentación/exportación sin
   llamar al núcleo analítico. Elegir evento individual para revisar geometría;
   el resumen muestra el proceso completo. Comparar eventos de distintos ciclos
   o ensayos desde la sección de comparación.

La interfaz conserva el modo **Legacy Calisto**. El editor de clasificación usa
`accepted_event`, `noise`, `instrumental_oscillation` y `pending_review`.
La propuesta automática queda separada y se conserva aunque el usuario la
clasifique de otra manera. La hipótesis de fase es un campo manual aparte.

## Terminal: tres acciones

Desde `/home/andrea/26_02_03`, con el entorno `.venv`:

```bash
# 1. Carga: proceso completo, sin truncar al PDF ni ajustar onsets.
.venv/bin/python cli.py explore \
  --format calvet --data data/raw/lvm/26_08_17.lvm \
  --program data/raw/pdf/26_08_17.pdf \
  --calvet-time sample --outdir out/common/calvet_signals

# 2. Dos ventanas editables de ejemplo, ambas pendientes de revisión.
.venv/bin/python cli.py analyze \
  --session out/common/calvet_signals \
  --windows data/event_selections/26_08_17_review.json \
  --outdir out/common/calvet_review

# 3. Geometría de un evento guardado; no recalcula el onset.
.venv/bin/python cli.py plot \
  --session out/common/calvet_review \
  --event 26_08_17:cooling:large-region \
  --font-size 18 --legend-size 14 --line-width 2 \
  --line-style=- --color '#554173' \
  --out out/common/large_region.pdf

# Proceso completo: omitir --event.
.venv/bin/python cli.py plot --session out/common/calvet_review \
  --out out/common/full_process.png
```

`--calvet-time sample` es una decisión explícita: usa el índice de muestras y
asume 0.5 s por muestra. Conserva el reloj original y sus diagnósticos; no prueba
sincronización con el PDF. Por defecto se usa el reloj registrado, que puede
contener segundos repetidos y hacer fallar un ajuste. No se aplica desfase ni
corrección térmica. Los huecos grandes o retrocesos del reloj original siguen
marcando discontinuidades aunque se elija un reloj reconstruido.

Otros lectores:

```bash
.venv/bin/python cli.py explore --format calisto \
  --data data/raw/lvm/23_05_12.txt --program data/raw/pdf/23_05_12.pdf \
  --outdir out/common/calisto_signals

.venv/bin/python cli.py explore --format trios --data /ruta/ensayo.xls \
  --outdir out/common/trios_signals
```

TRIOS conserva cada hoja como dataset: usar su nombre con `--dataset` al
graficar. Requiere unidades declaradas compatibles (s/min y Celsius); HF se
conserva sin conversión. Si faltan las unidades de HF, declararlas mediante
`--hf-unit W/g`, `mW/g` o `mW`, solo con respaldo instrumental. Si el reloj TRIOS
redondeado tiene duplicados, `--trios-time uniform` reconstruye explícitamente
el tiempo entre sus extremos; guarda ambos relojes y registra la suposición de
muestreo constante. No es una restauración verificada de tiempos individuales.
`--convention endo_positive|exo_positive|unknown` registra la convención aportada;
por defecto es desconocida. El signo geométrico positivo/negativo no se traduce
a endo/exo sin esa información.

### Ventanas manuales

El archivo `--windows` es una lista JSON, sin límite de dos eventos:

```json
[
  {
    "event_id": "run:cycle-1:event-1",
    "dataset_id": "run",
    "run_id": "run",
    "cycle_id": "cycle-1",
    "stage_id": "4",
    "window_s": [100, 300],
    "baseline_s": [110, 150],
    "flank_s": [180, 200],
    "polarity": "positive",
    "manual_classification": "pending_review",
    "phase_hypothesis": null
  }
]
```

Todos los tiempos son segundos del reloj elegido, no relativos a la etapa.
Los IDs deben ser únicos; `dataset_id` debe coincidir con un dataset cargado.
`baseline_s` y `flank_s` pueden omitirse o ser `null` para selección automática.
El flanco debe seguir a la línea base. `polarity` puede ser `auto`, `positive`
o `negative`. Cuando se omite, el algoritmo busca la primera desviación
persistente de cualquier signo, no el pico dominante.

`row_range: [inicio, fin]` permite seleccionar índices de filas canónicas
**inclusivos, desde cero**, antes del filtro temporal. Sirve para separar
tramos del archivo con reinicios de reloj. Los gráficos individuales y las
comparaciones respetan esas filas. Se conservan además los `source_row`
expuestos por cada lector; filas canónicas y filas del archivo no son siempre
lo mismo. Una discontinuidad de filas TXT/XLS omitidas bloquea el ajuste.

Para revisar límites/dirección de etapas por terminal, `analyze --stages
stages.json` acepta una lista:

```json
[{"dataset_id":"run","id":"4","start_s":100,"end_s":1000,
  "direction":"cooling","cycle_id":"cycle-1"}]
```

Sustituye solo esas etapas dentro de la nueva revisión y registra origen manual.
Un evento con `stage_id` debe estar dentro de la etapa. Los límites del PDF son
programados; su alineación con la adquisición requiere revisión. Por ejemplo,
el PDF de `26_08_17` llama a la etapa 2 `Inicio`: el adaptador deja su dirección
como `unknown` en vez de inventar calentamiento a partir del nombre.

### Propuestas automáticas y revisiones

```bash
.venv/bin/python cli.py analyze --session out/common/calvet_signals \
  --propose --stage 4 --direction cooling \
  --candidate-separation-s 600 --tangent-half-width-s 30 \
  --outdir out/common/candidates

# Editar candidates.json: añadir/quitar ventanas, ajustar regiones, clasificar.
.venv/bin/python cli.py analyze --session out/common/candidates \
  --windows out/common/candidates/candidates.json \
  --outdir out/common/reviewed_candidates
```

`candidates.json` no contiene eventos aceptados. `candidate_reports.json` guarda
método, configuración, limitaciones y motivos si no hubo propuestas. La
mediana móvil elimina tendencias lentas para **proponer** candidatos; se buscan
máximos locales del residuo absoluto con prominencia local en un radio de
`candidate_separation_s` y se aplica separación temporal mínima. No hay límite
fijo de eventos. Puede perder eventos anchos/débiles y proponer oscilaciones o
deriva; las ventanas manuales siguen funcionando si no propone nada.

## Método y diagnósticos

Los ajustes usan tiempo creciente en orden de adquisición; nunca se ordena por
T. En una isoterma sigue siendo posible ajustar el onset en tiempo. Cada ventana
produce una única intersección, o un motivo de fallo; no se generan dos
transiciones a partir de dos construcciones geométricas del mismo evento.

- Línea base automática: primer `baseline_fraction` (0.2) de la duración de la
  ventana. Línea base manual: el intervalo suministrado. Ajuste OLS de grado 1
  a HF original, centrando tiempo para estabilidad numérica.
- Flanco automático: primera desviación respecto a la línea base que supere
  `noise_multiplier × sigma` durante `min_persistence_s`. Se busca la máxima
  pendiente del signo elegido en la subida inicial hasta el primer máximo local;
  la recta se ajusta a HF original dentro de `tangent_half_width_s` a cada lado.
  El flanco manual usa exactamente su intervalo y evita la búsqueda automática.
- `smooth_s`, por defecto 0, aplica media móvil centrada **en tiempo** solo a la
  selección automática del flanco. Los ajustes y el gráfico conservan HF original.
- La intersección debe estar después del final de la línea base y antes del final
  del flanco, dentro del registro. Rectas paralelas, regiones insuficientes,
  tiempos no crecientes o huecos bloquean el resultado. No se fuerza un valor.
- Ruido: máximo entre desviación estándar y MAD escalado de residuos de línea
  base. SNR: máxima desviación absoluta posterior a la línea base en **esa ventana**
  dividida por ese ruido. La persistencia se calcula dentro de esa región y se
  informa en puntos y en segundos entre primera y última muestra de la racha.
- Calidad: RMSE y R² de ambas rectas; pendientes de las dos mitades de línea base
  y su diferencia. Sensibilidad del onset al quitar un extremo de cada región
  por vez; **no es un intervalo de confianza ni incertidumbre validada**.
- Posible oscilación: concentración espectral y cruces de cero tras quitar una
  tendencia lineal, ventana Hann, al menos tres ciclos. Requiere intervalos
  uniformes dentro del 10%. La concentración se mide en el bin dominante y sus
  dos vecinos. No identifica origen instrumental: eventos físicos repetidos,
  deriva y no estacionariedad pueden confundir el indicador.

Los umbrales son configurables, quedan guardados por evento y son criterios
operativos pendientes de validación experimental. Un resultado geométrico no
acepta automáticamente la región como evento ni le asigna una fase. Señales
manuales de baja amplitud se ajustan aunque no cumplan los umbrales automáticos.

T y P se interpolan entre filas contiguas registradas en el tiempo del onset.
No hay extrapolación, puente sobre valores faltantes, ni interpolación sobre un
intervalo superior a `max_gap_factor` (5) veces la mediana de pasos positivos.
`--max-gap-s` permite imponer un límite adicional más estricto. Las variables
no registradas quedan como no disponibles, con motivo. La temperatura genérica
de Calisto puede ser de horno; no se renombra como temperatura de muestra.
TRIOS conserva temperatura instrumental sin afirmar identidad de muestra.

## Guardado y comparación

Cada bundle contiene `results.json`, `windows.json`, señales CSV con hash SHA256
y copias exactas de las fuentes en `sources/`, con hashes, nombres y rutas de
origen. Guarda formato, unidades, convención, reloj, advertencias, etapas,
selecciones, parámetros, rectas, diagnósticos, revisión, hipótesis e interpolación.
Las revisiones referencian el bundle padre y copian fuentes archivadas: no
necesitan que siga existiendo el upload temporal. El replay verifica hash de las
señales. No se sobrescriben bundles/figuras sin `--overwrite`; una revisión debe
ir fuera del bundle padre. Nunca se permite reemplazar fuentes con un gráfico.

```bash
.venv/bin/python cli.py plot --session out/common/calvet_review \
  --compare out/common/calvet_review 26_08_17:cooling:large-region \
  --compare out/common/another_review another-event-id \
  --out out/common/comparison.svg
```

La comparación alinea en el onset guardado y muestra HF original contra tiempo
relativo; exige las mismas unidades y no normaliza amplitudes ni líneas base.
Permite revisar repetibilidad, sin declararla comprobada por superposición.

## Diferencias encontradas y alcance de compatibilidad

| Ruta histórica | Comportamiento observado | Decisión en el flujo común |
|---|---|---|
| `src/onset.py`, `src/event_detection.py`, `cli.py` | Onset en T; búsqueda alrededor de candidatos; selección de hasta dos; tiempo por T medida más cercana | Tiempo de adquisición, múltiples ventanas, interpolación local de variables |
| `src/trios_onset.py` | Solo primer positivo en enfriamiento; reloj uniforme reconstruido; umbral 20 MAD | Ambos signos, calentamiento/isoterma/enfriamiento, reconstrucción solo explícita |
| `process_dsc_data.py`, `util/calvet_analysis.py` | Selección cercana al pico; dos construcciones geométricas; método simple alternativo | Intersección única y regiones manuales, primer flanco para selección automática |
| `src/noise_analysis.py` | Persistencia podía venir de fuera del candidato | Corregido también en la ruta histórica; la máscara mantiene separadas regiones disjuntas |
| `app.py` | Calisto y resultados editados sin persistir revisión por evento | Flujo común con editor dinámico y revisión guardada; modo histórico conservado |
| `plot_calvet_results.py`, `src/replay.py` | Replay de esquemas específicos existentes | Se conservan; el renderer común lee el nuevo esquema, sin recalcular |

No se reemplazaron los algoritmos históricos porque cambiarían los resultados
existentes. Para comparar equipos con el mismo método, utilizar las tres acciones
comunes. `plot_time_profile.py` sigue siendo exploración gráfica: sus marcas de
onset suministradas por temperatura no equivalen a un ajuste calorimétrico.

## Verificación experimental y pendientes

Ver [common_calorimetry_review.md](common_calorimetry_review.md) para la revisión
local. Las pruebas sintéticas verifican comportamiento numérico, protección de
fuentes, unidades, intervalos y replay; no validan científicamente umbrales,
identidad de fases, incertidumbre, repetibilidad ni equilibrio termodinámico.

Quedan pendientes: validación experimental de criterios, calibración de relojes y
convenciones por equipo, incertidumbre con calibración instrumental y
correlación, separación de eventos realmente solapados y revisión experimental
con `.xls` TRIOS (no disponible en este checkout). El editor usa tablas numéricas,
no selección de regiones arrastrando sobre el gráfico. No se implementa conversión
mW ↔ W/g: falta una masa y base de normalización verificadas.
