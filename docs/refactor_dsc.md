# Diagnóstico y alcance de la refactorización

Calvet identifica el calorímetro y Calisto su software. La separación de lectores y métodos descrita aquí corresponde a los formatos LVM y TXT/PDF y a las señales que contienen, sin tratar Calisto como otro equipo.

## Acoplamientos encontrados

| Flujo | Situación inicial | Cambio / límite actual |
| --- | --- | --- |
| `process_dsc_data.py` | Mezclaba selección, métodos numéricos, representación y resumen; `plot_panel_6` calculaba onset al dibujar | Métodos movidos a `util/calvet_onset.py`; API importada compatible. La CLI usa `calvet_workflow.py`. El antiguo `plot_panel_6` se redirige al resumen temporal compartido y no calcula eventos. |
| LVM | `util/lvm_io.py` reconstruía 0,5 s/fila aunque guardaba `datetime`; recortaba por programa | API histórica conserva su valor predeterminado. Nueva CLI elige tiempo registrado, documenta ambas bases y conserva fila fuente/unidades/proveniencia. |
| Etapas | Selección PDF con rebasing; fallback por derivada del horno | `util/calvet_stages.py` distingue programa y condiciones medidas, mantiene el origen de corrida y permite límites manuales. No transforma una etapa programada en una condición efectivamente alcanzada. |
| Eventos Calvet | Un onset durante la representación de cada zona; marcadores manuales a temperaturas llamadas onsets | `util/calvet_events.py` admite varias ventanas y resultados no determinados. Los objetivos de temperatura legacy se documentan como observaciones manuales. |
| Gráficos | El panel de seis era la salida principal | `util/calvet_plotting.py` usa como salida principal tres paneles con tiempo compartido: presión, temperatura de muestra/Huber disponible y HF. El panel de seis está retirado también de `analisis.py`; los accesos antiguos apuntan al resumen temporal. P–T, HF y comparaciones siguen como salidas explícitas desde resultados guardados. |
| Metadata | CSV carga/corrida sin vínculo al resultado ni documentación extensa | El ejecutor transmite IDs y JSON opcional; `util/experimental_metadata.py` guarda originales, hashes, parámetros y pendientes. No inventa composiciones. |
| Calisto | Lectores/métodos ya separados, pero `cli.analyze` siempre graficaba; detector asignaba TLL/TSL por temperatura; UI ignoraba límites editados | `make_plots=False`, persistencia de segmentos/eventos y `--render`; hipótesis de fase optativa, confianza de asignación separada. La UI aplica límites y guarda revisiones aparte. |
| `compare_pt_path_nist.py` | Referencias de CO₂ puro rotuladas genéricamente; onset fijo | Rótulos explícitos de CO₂ puro. El onset fijo se conserva por compatibilidad y se explica en README; no está conectado al cálculo nuevo. |
| `analyze_step_hf.py` | Lectura/derivadas, segmentación, líneas base y relajación, gráficos en un script | Inspeccionado y conservado como diagnóstico térmico/instrumental. No se mezclaron sus residuos HF con métodos de transición ni con Calisto. |

## Trazabilidad y supuestos

La identificación automática de layouts LVM utiliza heurísticas existentes de rangos y variación. Se registra cuál se eligió y que su confirmación instrumental está pendiente. La asignación HF_mW/HF_uV y temperatura Huber se conserva para el layout de presión después de HF. Los nombres/unidades provienen del contrato histórico del lector; no sustituyen una certificación de calibración.

El origen de las zonas PDF sigue suponiendo coincidencia con el inicio de registro. No hay evidencia para aplicar un desfase automático. Se conserva esta advertencia y la edición manual de límites. La corrección de retraso térmico no se ha implementado ni confundido con el alineamiento temporal. Temperatura de muestra, horno y Huber son columnas independientes. No existe consigna registrada en el LVM analizado; no se fabrica una.

El tiempo registrado tiene segundos repetidos. No se inventan fracciones de segundo: el análisis nuevo utiliza la primera fila de cada segundo y lo documenta. Para comparar con el algoritmo histórico se selecciona explícitamente `legacy`. Ambas opciones conservan tiempo original. Los ajustes nuevos rechazan interrupciones y las interpolaciones sólo usan vecinos válidos dentro del límite configurado. Los lectores antiguos que aún usan la base histórica requieren revisar ese supuesto para análisis nuevos.

La velocidad medida es una regresión de temperatura de muestra frente a tiempo sobre la selección, expresada en °C/min. No demuestra estabilización, uniformidad térmica ni equilibrio. El parser Calvet de duración/etiqueta no extrae todavía todas las consignas y velocidades: esas magnitudes quedan pendientes.

La relación carga/corrida/etapa/evento se guarda con cada evento. Si un evento cruza varias etapas y no se proporciona una etapa inequívoca, su pertenencia queda pendiente con advertencia. Los campos desconocidos no impiden gráficos exploratorios. No se calculan masas, composiciones ni entalpías a partir de supuestos no verificados.

En el flujo TXT/PDF de Calisto se conserva el ajuste en temperatura del detector existente. El campo histórico `peak_temp_c` apunta al candidato localizado por gradiente y se documenta en el resultado. El tiempo de onset sólo se obtiene cuando hay un único cruce temporal contiguo de la temperatura calculada; una trayectoria ambigua queda sin tiempo asignado. El lector TXT actual no incorpora presión. El algoritmo de máximo temporal usado con LVM permanece separado de este detector.

## Comprobación numérica

`tests/fixtures/calvet_legacy_26_08_17.json` contiene resultados calculados con el código anterior a la extracción, obtenido de HEAD antes de modificarlo. Usa el archivo SHA-256 indicado, tiempo histórico, intervalo 38,5–40,7 h y una de cada 20 filas para mantener manejable el coste del buscador automático de línea base. Se conservan tangente, onset2, pico, coeficientes, ventanas y umbral. Se comparan con tolerancias `rtol=1e-12`, `atol=1e-12`: adecuadas para las mismas operaciones de doble precisión sobre la misma selección, sin expresar incertidumbre experimental.

La extracción mantuvo las operaciones numéricas; los valores predeterminados se centralizaron. Cambiar ventana, cadencia, origen o parámetros puede cambiar resultados. La regresión prueba conservación de implementación sobre una selección conocida, sin afirmar equivalencia de todos los experimentos ni validación científica de onsets.

## Pendientes explícitos

- Confirmar calibración/unidades instrumentales, masas, ambos métodos de masa de CO₂, composiciones vinculadas a cada método, volumen y configuración de E24.
- Revisar las ventanas exploratorias y contrastar LL?/S? con evidencia experimental. La imagen no aporta esa validación.
- Extraer y revisar velocidades/consignas completas del programa Calvet y relacionar el programa Huber. El ciclo Calvet sigue siendo un ordinal de dirección, no un agrupamiento físico validado.
- La edición de ventanas Calisto en UI utiliza JSON explícito y su metadata de carga se admite por CLI. Una interfaz gráfica de selección de ventanas y formulario de carga sería una mejora futura; las ventanas independientes ya se analizan y persisten.
- Migrar gradualmente los restantes scripts diagnósticos y las rutas antiguas de P–T a las nuevas carpetas/configuraciones cuando corresponda. `analyze_step_hf.py`, TRIOS y scripts auxiliares conservan sus contratos y salidas previas; el nuevo comportamiento de no sobrescritura está aplicado al flujo principal Calvet y a la CLI/UI Calisto.
- Integrar revisiones manuales tabulares de Calisto con una próxima ejecución sólo mediante una decisión explícita. Hoy se guardan aparte para preservar el resultado automático original.

En `26_08_17.lvm`, el reloj muestra un salto de 60 s entre 208 y 268 s. La base histórica oculta ese corte: también se comprueba el reloj original al analizar o interpolar usando `legacy`. Las ventanas exploratorias guardadas para E24 no atraviesan ese intervalo.

Se corrigió además un error de lectura Calisto demostrado con `23_05_12.txt`: el tiempo `1,002` se interpretaba como 1,002 s, mientras `1,001.7` se interpretaba como 1001,7 s. El lector ahora infiere por columna la convención de miles/decimales a partir de valores inequívocos con ambos separadores y la aplica también a filas enteras. Se registra el criterio y se prueba con datos representativos. Esta reparación puede cambiar la asignación de etapas; no se afirma conservación de selecciones producidas por el parser defectuoso. Las fórmulas de los onsets automáticos Calisto se conservan y la prueba de heurística compara los mismos datos de entrada.
