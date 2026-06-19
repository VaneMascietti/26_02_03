# 26_02_03 - Analisis DSC / Calvet

Scripts para analizar ensayos `.lvm` y generar:
- Panel de 6 graficos por ensayo o por zona (`process_dsc_data.py`)
- Grafico de fase CO2 `P vs T` con isocoras NIST (`compare_pt_path_nist.py`)

## Dónde Poner Datos

- Inputs crudos LVM: `data/raw/lvm/`
- Inputs crudos PDF: `data/raw/pdf/`
- Compatibilidad legacy: si no existe en `data/raw/...`, el codigo busca en `data/`.

Ejemplos:
- `data/raw/lvm/26_01_30_teste_oring_nitrilica.lvm`
- `data/raw/pdf/26_01_30_teste_oring_nitrilica.pdf`

## Requisitos

- Python 3.10+
- Paquetes: `numpy`, `pandas`, `matplotlib`, `PyPDF2`

Instalacion rapida:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib PyPDF2
```

Alternativa usando archivos versionados:

```bash
pip install -r requirements.txt
```

## Estructura

- `process_dsc_data.py`: carga `.lvm`, parsea zonas del PDF del programa, genera panel 6 y calcula onset/onset2.
- `compare_pt_path_nist.py`: superpone trayectoria experimental con curvas NIST (isocoras y saturacion).
- `data/`: experimentos `.lvm` y programas `.pdf`.
- `nist/`: tablas NIST locales.
- `out/figures/`: figuras generadas.
- `out/tables/`: tablas generadas.
- `out/reports/`: reportes generados.

## Uso de `process_dsc_data.py`

Comando base:

```bash
.venv/bin/python process_dsc_data.py --data <archivo.lvm> --program <programa.pdf>
```

Ejemplo por zona (zona 3):

```bash
.venv/bin/python process_dsc_data.py \
  --data 26_01_30_teste_oring_nitrilica.lvm \
  --program 26_01_30_teste_oring_nitrilica.pdf \
  --zone 3
```

Seleccion por perfil de zona:

```bash
# segundo Cooling del programa
.venv/bin/python process_dsc_data.py --data <archivo.lvm> --program <programa.pdf> --Cooling 2
```

Parametros utiles de onset:

```bash
# metodo de tangente (default)
--onset-method tangent

# metodo simple por umbral
--onset-method simple

# forzar ventana de pico mas cerca del pico (menor valor = mas cerca)
--flank-near-peak-frac 0.10
```

Salida esperada:
- Imagen: `out/figures/process_dsc_data/<stem>_panel_6plots_zone<Z>.png` (si se usa zona)
- Consola: valores de onset, onset2, y ventanas usadas:
  - `base=[t0,t1] h`
  - `pico=[t0,t1] h`

## Uso por metadata experimental

Para experimentos divididos en varias corridas, usar:

```bash
.venv/bin/python process_experiments_from_metadata.py --metadata data/metadata/experiments.csv
```

Cada fila de `data/metadata/experiments.csv` representa un archivo `.lvm`. Las columnas minimas son:

```csv
experiment_id,load_id,run_id,date,stage,file_lvm,file_program,notes
```

`experiment_id` agrupa el experimento completo, `load_id` identifica la misma carga/celda sellada y `run_id` identifica una corrida individual. Asi, dos dias diferentes pueden pertenecer a la misma carga:

```csv
E08,L08,R01,2026-03-30,heating,26_03_30.lvm,26_03_30.pdf,heating da mesma carga
E08,L08,R02,2026-03-31,cooling,26_03_31.lvm,26_03_31.pdf,cooling da mesma carga
```

Ejemplos utiles:

```bash
# Ver que comandos se ejecutarian para una carga
.venv/bin/python process_experiments_from_metadata.py --load-id L08 --dry-run

# Procesar solo cooling
.venv/bin/python process_experiments_from_metadata.py --stage cooling

# Para usar automaticamente --all-heating o --all-cooling segun stage
.venv/bin/python process_experiments_from_metadata.py --load-id L08 --select-stage-zones

# Pasar argumentos extra a process_dsc_data.py
.venv/bin/python process_experiments_from_metadata.py --load-id L08 -- --onset-method simple
```

## Stepwise cooling HF analysis

Para enfriamientos escalonados tipo rampa-isoterma-rampa-isoterma en Calvet HP, usar:

```bash
python analyze_step_hf.py \
  --data-heating data/raw/lvm/26_05_12.lvm \
  --program-heating data/raw/pdf/26_05_12.pdf \
  --data-cooling data/raw/lvm/26_05_13.lvm \
  --program-cooling data/raw/pdf/26_05_13.pdf \
  --exclude-transition-window 25 30 \
  --outdir out/figures/step_hf \
  --transparent \
  --diagnostic-note "N2 cylinder closed during experiment"
```

Este flujo trata el HF como diagnostico termico/instrumental. No interpreta automaticamente onsets LL, no integra entalpia y no reporta `transition enthalpy`. Genera figuras en `out/figures/step_hf/` y tablas en `out/tables/step_hf/`, incluyendo:

- `*_zone_summary.csv`
- `*_isotherm_relaxation.csv`
- `*_step_hf_processed.csv`

## Uso de `compare_pt_path_nist.py`

Ejecucion con argumentos:

```bash
.venv/bin/python compare_pt_path_nist.py \
  --data 26_02_07_teste_oring_vitom.lvm \
  --program 26_02_07_teste_oring_vitom.pdf \
  --label "Vitom" \
  --nist-iso nist/nist_iso_D0p70000.txt \
  --nist-iso nist/nist_iso_D0p80000.txt \
  --nist-sat nist/co2_nist_sat.csv \
  --out out/figures/compare_pt_path_nist/co2_phase_overlay.png
```

Opcional:
- `--show-onset`: muestra punto de onset de referencia en el grafico
- `--no-sat`: no dibuja curva de saturacion NIST
- `--data2` y `--label2`: segunda trayectoria experimental

Salida esperada:
- Imagen: `out/figures/compare_pt_path_nist/co2_phase_overlay.png` (o el archivo indicado en `--out`)

## Notas

- `process_dsc_data.py` soporta dos formatos de `.lvm`:
  - Historico (9 columnas)
  - Nuevo (10 columnas, con `P_referencia` en penultima columna)
- Si la zona no existe en el PDF, el script informa las zonas disponibles.
- Se recomienda ejecutar siempre con `.venv/bin/python` para evitar problemas de dependencias.

## Testing

Instalar dependencias de desarrollo:

```bash
pip install -r requirements-dev.txt
```

Correr tests:

```bash
pytest -q
```

Chequeo de sintaxis:

```bash
python -m py_compile process_dsc_data.py compare_pt_path_nist.py util/path.py
```
