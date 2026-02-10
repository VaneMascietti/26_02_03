# 26_02_03 - Analisis DSC / Calvet

Scripts para analizar ensayos `.lvm` y generar:
- Panel de 6 graficos por ensayo o por zona (`analisis.py`)
- Grafico de fase CO2 `P vs T` con isocoras NIST (`co2_phase_plot.py`)

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

- `analisis.py`: carga `.lvm`, parsea zonas del PDF del programa, genera panel 6 y calcula onset/onset2.
- `co2_phase_plot.py`: superpone trayectoria experimental con curvas NIST (isocoras y saturacion).
- `data/`: experimentos `.lvm` y programas `.pdf`.
- `nist/`: tablas NIST locales.
- `output/analisis/`: salidas de panel 6.
- `output/phase/`: salidas de grafico de fase CO2.

## Uso de `analisis.py`

Comando base:

```bash
.venv/bin/python analisis.py --data <archivo.lvm> --program <programa.pdf>
```

Ejemplo por zona (zona 3):

```bash
.venv/bin/python analisis.py \
  --data data/26_01_30_teste_oring_nitrilica.lvm \
  --program data/26_01_30_teste_oring_nitrilica.pdf \
  --zone 3
```

Seleccion por perfil de zona:

```bash
# segundo Cooling del programa
.venv/bin/python analisis.py --data <archivo.lvm> --program <programa.pdf> --Cooling 2
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
- Imagen: `output/analisis/<stem>_panel_6plots_zone<Z>.png` (si se usa zona)
- Consola: valores de onset, onset2, y ventanas usadas:
  - `base=[t0,t1] h`
  - `pico=[t0,t1] h`

## Uso de `co2_phase_plot.py`

Ejecucion con argumentos:

```bash
.venv/bin/python co2_phase_plot.py \
  --data data/26_02_07_teste_oring_vitom.lvm \
  --program data/26_02_07_teste_oring_vitom.pdf \
  --label "Vitom" \
  --nist-iso nist/nist_iso_D0p70000.txt \
  --nist-iso nist/nist_iso_D0p80000.txt \
  --nist-sat nist/co2_nist_sat.csv \
  --out output/phase/co2_phase_overlay.png
```

Opcional:
- `--show-onset`: muestra punto de onset de referencia en el grafico
- `--no-sat`: no dibuja curva de saturacion NIST
- `--data2` y `--label2`: segunda trayectoria experimental

Salida esperada:
- Imagen: `output/phase/co2_phase_overlay.png` (o el archivo indicado en `--out`)

## Notas

- `analisis.py` soporta dos formatos de `.lvm`:
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
python -m py_compile analisis.py co2_phase_plot.py
```
