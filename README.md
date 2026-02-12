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
