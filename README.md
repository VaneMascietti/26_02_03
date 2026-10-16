# Análisis DSC / Calvet

El flujo principal selecciona **carga → corrida → etapa → eventos** y guarda resultados antes de graficarlos. `26_08_17.lvm` corresponde a E24, CO₂ + n-C16, y `Fig5.png` es la referencia confirmada por el usuario. Los archivos crudos se leen sin modificarlos.

Calvet es el calorímetro y Calisto es su software. Los flujos de lectura se distinguen por el formato de exportación y las señales disponibles: LVM con programa PDF, o TXT/PDF exportados desde Calisto. Se conservan los métodos y supuestos compatibles con cada formato. TRIOS tiene otro lector.

## Instalación y pruebas

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
MPLBACKEND=Agg .venv/bin/python -m pytest -q
```

Los datos se buscan primero en `data/raw/lvm/` y `data/raw/pdf/`, después en `data/`. Los comandos siguientes se ejecutan desde la raíz del proyecto. Se recomienda `.venv/bin/python`.

## Calvet HP: comandos

Cada ejecución crea una carpeta nueva en `out/analysis/calvet/` con huella de configuración y fecha UTC. Allí quedan `signals.csv`, `results.json`, `config.json`, `warnings.txt` y las figuras solicitadas. La configuración incluye SHA-256 de los archivos originales. Las advertencias de metadata incompleta permiten continuar el análisis exploratorio.

### 1. Procesar y revisar sincronización

```bash
MPLBACKEND=Agg .venv/bin/python process_dsc_data.py \
  --data 26_08_17.lvm --program 26_08_17.pdf \
  --metadata data/metadata/E24_26_08_17.json
```

La figura predeterminada (`signals.png`) tiene tres paneles alineados sobre el mismo eje de tiempo: presión de muestra, temperatura de muestra junto con Huber cuando esté disponible, y heat flow. La temperatura de horno queda conservada en los datos. El panel de seis gráficos fue retirado. `--output pt` agrega una trayectoria por etapas si se solicita; las otras salidas independientes disponibles son `hf-time` y `hf-temperature`. Las salidas se pueden repetir. `--no-plots` ejecuta solamente el procesamiento y análisis.

La base predeterminada usa las fechas registradas (`--time-basis recorded`). `t_original_s` conserva el tiempo relativo del reloj exportado; `t_sync_s` es el tiempo elegido y `t_h=t_sync_s/3600`. Las señales comparten el tiempo de cada fila: **no se estima ni aplica un desfase**. Las fechas tienen resolución de un segundo y aparecen duplicadas. Para ajustar eventos se conserva la primera fila de cada segundo, con advertencia; `signals.csv` conserva todas las filas seleccionadas.

`--time-basis legacy` reproduce la cadencia histórica supuesta de 0,5 s/fila (`t_acquisition_s`), necesaria para comparaciones con resultados antiguos. El diagnóstico informa diferencias entre ambas bases, repeticiones, retrocesos y saltos. Cambiar la base temporal puede cambiar límites y onsets; no es una corrección del retraso térmico instrumental. No se realiza esa corrección.

### 2. Seleccionar una etapa y corregir límites

```bash
MPLBACKEND=Agg .venv/bin/python process_dsc_data.py \
  --data 26_08_17.lvm --program 26_08_17.pdf \
  --metadata data/metadata/E24_26_08_17.json \
  --Cooling 1 --output pt
```

También se conservan `--zone 4`, `--Heating N`, `--Isoterma N`, `--Return N`, `--all-cooling` y `--all-heating`. `--stage cooling --cycle 1` selecciona el primer enfriamiento; aquí ciclo significa **ordinal dentro de la dirección seleccionada**, registrado explícitamente. Para seleccionar corridas/cargas se usa el CSV experimental descrito más abajo.

El PDF de E24 llama `Inicio` a la rampa ascendente: se conserva esa etiqueta. `--stage heating --cycle 1` permite seleccionarla por su dirección medida. Se guardan por separado tipo programado, dirección y velocidad medidas sobre `T_muestra`. La velocidad programada permanece pendiente cuando el parser Calvet no la extrae; no se infiere a partir de la medida.

`--time-range 31 35` recorta un intervalo en horas desde el inicio de la corrida, sin cambiar su origen. `--Cooling 1 --stage-limits 22.4 40.8` corrige manualmente los límites de esa etapa, conserva los del PDF y permite incluir datos más allá del fin previsto. La correspondencia entre inicio del programa e inicio del registro requiere revisión experimental.

Por compatibilidad, sin `--full-data` se recorta a la duración total del PDF. `--full-data` conserva todo el registro y no se combina con selección de zonas. `--stage-limits` carga todo el registro antes de seleccionar la etapa corregida.

### 3. Analizar uno o varios eventos, sin figuras

```bash
MPLBACKEND=Agg .venv/bin/python process_dsc_data.py \
  --data 26_08_17.lvm --program 26_08_17.pdf \
  --metadata data/metadata/E24_26_08_17.json \
  --Cooling 1 --events data/metadata/E24_events.example.json --no-plots
```

El ejemplo analiza dos ventanas independientes de un mismo enfriamiento. Son ventanas exploratorias; requieren revisión. Un evento débil puede quedar `undetermined` con su motivo. No se crea obligatoriamente un evento por etapa.

Para un evento individual se puede usar:

```bash
MPLBACKEND=Agg .venv/bin/python process_dsc_data.py \
  --data 26_08_17.lvm --program 26_08_17.pdf \
  --metadata data/metadata/E24_26_08_17.json --Cooling 1 \
  --event-id feature_low --event-window 38.5 40.6 \
  --baseline-window 38.5 39 --flank-window 39.5 39.65 \
  --onset-method tangent --no-plots
```

Las ventanas nuevas son absolutas en horas de corrida. `--onset-method simple` conserva el umbral histórico; sus parámetros se pasan mediante `params` en el JSON y no admite ventanas de ajuste de rectas. `--onset-around` y `--onset-window` siguen disponibles: si se selecciona una zona con las opciones antiguas, el centro se interpreta relativo a ella, salvo `--no-rebase`. Ahora esa ventana también limita la búsqueda tangente; antes el panel la ignoraba para tangente.

Cada evento guarda identificadores, señales/unidades, parámetros efectivos, ventanas, coeficientes de ajuste, onset y pico, temperatura y presión simultáneas, advertencias y revisión manual. La presión indica `measured`, `interpolated` o `unavailable`. La interpolación utiliza solamente filas adyacentes válidas, sin extrapolación ni saltos mayores de `--max-gap-s` (5 s por defecto). No se ajusta un evento atravesando interrupciones o selecciones discontinuas.

**`onset2` es una segunda estimación del mismo evento:** intersección de la línea base con la regresión lineal del flanco. `onset` usa la tangente en el punto de mayor pendiente. Se conserva el nombre histórico para comparar resultados; no representa otra transición.

### 4. Graficar resultados guardados sin recalcular

El comando de análisis imprime la ruta exacta de `results.json`. En este ejemplo se elige el último archivo de la corrida:

```bash
RESULTS=$(ls -td out/analysis/calvet/26_08_17_* | head -n 1)/results.json
MPLBACKEND=Agg .venv/bin/python plot_calvet_results.py "$RESULTS" --plot hf-time --event-id feature_low
MPLBACKEND=Agg .venv/bin/python plot_calvet_results.py "$RESULTS" --plot hf-temperature --event-id feature_low
MPLBACKEND=Agg .venv/bin/python plot_calvet_results.py "$RESULTS" --plot pt
```

El renderizador necesita únicamente `results.json` y su `signals.csv`, sin LVM ni PDF. No calcula onsets. Cada nuevo gráfico evita sobrescribir los anteriores. Para comparar dos selecciones guardadas:

```bash
.venv/bin/python plot_calvet_results.py ruta1/results.json ruta2/results.json \
  --plot compare --out out/figures/comparacion_e24.png
```

La comparación superpone HF frente a temperatura (`--x T_muestra`) o tiempo (`--x t_h`), con los identificadores de carga/corrida. Elegí resultados de las etapas o ciclos de interés. Si el destino existe, se solicita otra ruta mediante un error explícito.

### 5. Revisar presión, temperaturas y heat flow frente al tiempo

```bash
MPLBACKEND=Agg .venv/bin/python plot_calvet_results.py "$RESULTS" --plot signals
```

Genera presión, temperatura de muestra/Huber y HF frente al tiempo, sin calcular onsets. Huber se omite si no hay una señal con valores válidos. Los accesos antiguos `--diagnostic-panel`, `--output diagnostic`, `--plot diagnostic` y las funciones `plot_panel_6()` de los scripts principales ahora generan esta misma figura de tiempo. No generan el panel de seis.

Los accesos antiguos `--plot-pt`, `--plot-pt-time`, `--data2`, `--label2` siguen disponibles para trayectorias completas y crean su propia carpeta identificada. Para selección por etapa utilizá `--output pt`.

## Metadata y criterio científico

`data/metadata/E24_26_08_17.json` documenta únicamente la identidad y componentes confirmados. Masas, volumen, composición, programa Huber e incertidumbres siguen pendientes. Una lista de incidencias vacía significa que no fueron documentadas.

El JSON admite `loaded_masses`, `co2_mass_determinations` (varios resultados con `method_id`, valor, unidad y fuente) y `compositions` (cada composición con `mass_method_id`, fracciones molares/másicas y componentes). Se conservan ambos métodos de masa si existen; no se calcula una composición automáticamente ni se inventan incertidumbres. Se incluyen configuración del sistema, programas, incidencias y revisiones manuales. Los originales y parámetros quedan vinculados en cada resultado.

La tabla existente sigue funcionando:

```bash
.venv/bin/python process_experiments_from_metadata.py --load-id L08 --dry-run
.venv/bin/python process_experiments_from_metadata.py --load-id L08 --select-stage-zones -- --no-plots
```

Columnas mínimas: `experiment_id,load_id,run_id,date,stage,file_lvm,file_program,notes`. Una fila identifica una corrida; varias corridas pueden pertenecer a la misma carga. La columna opcional `metadata_json` enlaza su documentación ampliada. Los identificadores se transmiten al análisis.

Se distinguen observación, cálculo e interpretación física. La detección de un máximo/mínimo o cambio de pendiente no asigna LL, LV, LLV ni sólido. La confianza de detección es independiente de la asignación de fase. Un onset calorimétrico no valida un equilibrio. Las etiquetas LL?/S? de Fig5 se conservan como hipótesis pendientes. No se calculan entalpías.

## Flujos conservados

### Exportaciones TXT/PDF de Calisto

```bash
MPLBACKEND=Agg .venv/bin/python cli.py \
  --pdf data/23_05_12.pdf --txt data/23_05_12.txt \
  --max-cycles 1 --no-plots --outdir out/dsc_analysis_23_05_12
```

Guarda segmentos y eventos en `calisto_results.json`, una fila por evento en `events.csv`, más tablas y reporte existentes. Agregá `--cycle N`, `--zone N` o `--direction heating` para seleccionar. `--phase-heuristic` habilita la hipótesis histórica LL/SL según orden; por defecto los eventos son genéricos. La rama de ajuste de dos candidatos a baja temperatura se conserva numéricamente, sin asignar fases por defecto.

```bash
.venv/bin/python cli.py --render ruta/calisto_results.json
.venv/bin/streamlit run app.py
```

La aplicación permite corregir límites de zona (segundos), seleccionar dirección/ciclo, introducir ventanas independientes mediante JSON y guardar una revisión manual independiente. La opción `--events archivo.json` acepta `events` con `stage_id=zoneN`, `event_id`, `interval_s`, `baseline_s` y `flank_s`, todos en segundos desde el inicio de corrida. Usa la convención existente de ajuste robusto en temperatura para cada ventana; no presupone que el método Calvet sea compatible. `--metadata archivo.json` vincula la carga y corrida. La API Python `analyze(..., make_plots=False)` analiza sin figuras. El parser reconoce el agrupamiento de miles en tiempos como `1,001.7` y `1,002`, evitando retrocesos artificiales; esta corrección puede cambiar las asignaciones de zona respecto del lector antiguo. La CLI y la aplicación crean carpetas identificadas; la API rechaza sobrescribir un análisis existente.

El lector conserva temperatura de muestra y horno por separado, prefiere muestra y avisa si utiliza horno. Conserva el tiempo original y convertido, y convierte unidades conocidas de HF a mW; rechaza unidades incompatibles como W/g. El lector de estas exportaciones TXT no incorpora presión: queda explícitamente no disponible. `peak_temp_c` del detector conserva el significado histórico de candidato por gradiente, sin certificar un máximo calorimétrico.

### Calvet escalonado: diagnóstico HF

`analyze_step_hf.py` conserva su carácter diagnóstico. No asigna LL ni integra entalpías. Ejemplo de uso existente:

```bash
.venv/bin/python analyze_step_hf.py \
  --data-heating data/raw/lvm/26_05_12.lvm --program-heating data/raw/pdf/26_05_12.pdf \
  --data-cooling data/raw/lvm/26_05_13.lvm --program-cooling data/raw/pdf/26_05_13.pdf \
  --exclude-transition-window 25 30 --outdir out/figures/step_hf \
  --diagnostic-note "N2 cylinder closed during experiment"
```

### Referencias NIST y TRIOS

`compare_pt_path_nist.py` conserva sus opciones `--nist-iso`, `--nist-sat`, `--data2` y `--no-sat`. Sus curvas y puntos NIST se identifican como **referencias de CO₂ puro**, también cuando el experimento es una mezcla. `--show-onset` sigue siendo el punto fijo de referencia legacy, no un onset leído del resultado nuevo.

```bash
.venv/bin/python trios_cli.py --xls "data/100724_etoh_nc16_0.851w_0.2rate.xls" --outdir out/trios_100724_etoh_nc16
```

El flujo TRIOS sigue separado y conserva su lector Excel y onset temporal propios.

## Alcance y validación

El diagnóstico de responsabilidades, límites de compatibilidad y pendientes está en [docs/refactor_dsc.md](docs/refactor_dsc.md). Se comparan tangente, regresión del flanco y umbral con resultados previos sobre datos reales y se prueban unidades, selección, presión, interrupciones y reproducción guardada. Los archivos instrumentales no se distribuyen en git; las pruebas con archivos reales se omiten cuando faltan esos datos locales. Las pruebas de software no constituyen validación científica.
