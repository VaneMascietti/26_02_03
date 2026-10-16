# Análise de dados DSC/Calvet

Ferramentas em Python para analisar dados de calorimetria DSC/uDSC Calisto e
experimentos Calvet exportados em `.lvm`. O repositório reúne parsers,
detecção de eventos, análise de transições, visualizações e relatórios.

## Conteúdo do repositório

```text
.
├── app.py, cli.py                 # Aplicativo web e CLI para dados Calisto
├── src/                           # Parsers, detecção, incerteza, gráficos e relatórios
├── util/                          # Leitura LVM, zonas térmicas e utilitários compartilhados
├── process_dsc_data.py            # Análise de LVM e programa térmico em PDF
├── process_experiments_from_metadata.py
├── analyze_step_hf.py              # Análise diagnóstica de fluxo de calor por etapas
├── dynamic_hf_deconvolution.py     # Deconvolução de sinal de fluxo de calor
├── compare_pt_path_nist.py         # Trajetória P-T comparada com dados NIST
├── plot_*.py                       # Gráficos de trajetórias, transições e perfis temporais
├── data/
│   ├── metadata/                   # Metadados dos experimentos
│   ├── phase_transitions/          # Pontos experimentais e da literatura
│   ├── plot_sets/                  # Listas CSV usadas pelos fluxos de gráficos
│   └── literature/                 # Dados e referências bibliográficas
├── nist/                           # Tabelas locais de referência do CO2
├── outputs/                        # Exemplo versionado de análise Calisto
└── tests/                          # Testes automatizados
```

Arquivos de entrada brutos dos instrumentos não são incluídos no repositório.
Coloque-os localmente em `data/raw/lvm/` (`.lvm`) e `data/raw/pdf/` (programas
térmicos em PDF). Essa pasta é ignorada pelo Git. Os CSVs em `data/plot_sets/`
apontam para esses caminhos locais. A leitura de caminhos antigos diretamente
em `data/` ainda é aceita por compatibilidade.

Os resultados produzidos pelos scripts são gravados normalmente em `out/`,
também ignorado pelo Git. Já `outputs/dsc_analysis_23_05_12/` é um exemplo
versionado e não o diretório padrão para novas execuções.

## Requisitos e instalação

O código requer Python 3.10 ou superior. A integração contínua testa Python 3.11
e 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` instala as dependências da aplicação e do desenvolvimento
(incluindo pytest). Para instalar apenas as dependências de execução, use
`requirements.txt`.

## Flujo común de calorimetría (TRIOS, Calisto y Calvet)

`cli.py explore`, `cli.py analyze` y `cli.py plot` separan carga, selección de
múltiples eventos y gráficos de resultados guardados. `app.py` ofrece las mismas
acciones en **Common calorimetry**, con ventanas editables y revisión manual.
Los comandos históricos se conservan; sus métodos no son equivalentes al nuevo.

La [guía completa](docs/common_calorimetry.md) incluye comandos por formato,
esquema de ventanas, criterios de ruido/oscilación, método geométrico,
compatibilidad y limitaciones. La [revisión local](docs/common_calorimetry_review.md)
distingue pruebas sintéticas de observaciones experimentales.

```bash
.venv/bin/streamlit run app.py
.venv/bin/python cli.py explore --help
.venv/bin/python cli.py analyze --help
.venv/bin/python cli.py plot --help
```

## Fluxos principais

### Análise de experimentos Calvet HP em LVM/PDF

```bash
.venv/bin/python process_dsc_data.py \
  --data data/raw/lvm/26_03_30.lvm \
  --program data/raw/pdf/26_03_30.pdf \
  --zone 2 \
  --stage-window 6 6.740833333333334 \
  --event-window 0 0.740833333333334 \
  --load-id L08 \
  --run-id R01 \
  --results-out out/calvet
```

O fluxo Calvet HP lê o LVM e o programa PDF, permite selecionar a zona do
programa (`--zone`, `--Cooling`, `--Heating`, `--Isoterma` ou `--Return`) e
corrigir os limites da etapa em horas com `--stage-window`. `--event-window`
seleciona uma janela de evento independente, também em horas relativas à etapa
selecionada. Os dados de entrada não são alterados.

O processamento de eventos não gera figuras por padrão. Cada execução com
resultados é gravada em um novo subdiretório de `out/calvet/`, contendo sinais,
eventos, parâmetros, unidades, fontes, limites programados e metadados de tempo.
O JSON registra os relógios original, de amostras e do dispositivo, além dos
diagnósticos; nenhum desfase temporal automático nem correção de atraso térmico
é aplicado. Uma etapa programada não prova que a condição foi efetivamente
atingida. Se faltarem dados, o evento fica como não determinado com o motivo.

Para revisar sincronização/operação, selecione o modo `signals` e o relógio
desejado. Para reproduzir a trajetória P–T ou o sinal HF, use o mesmo bundle:

```bash
.venv/bin/python plot_calvet_results.py \
  --results out/calvet/<bundle>/results.json \
  --mode signals \
  --time-axis selected \
  --out out/figures/calvet-signals.png

.venv/bin/python plot_calvet_results.py \
  --results out/calvet/<bundle>/results.json \
  --mode pt \
  --out out/figures/calvet-pt.png

.venv/bin/python plot_calvet_results.py \
  --results out/calvet/<bundle>/results.json \
  --mode hf-temperature \
  --event-id <event_id> \
  --out out/figures/calvet-event.png
```

Esses gráficos usam os resultados guardados e não recalculam onsets. Os modos
`hf-time` e `hf-temperature` aceitam eventos individuais, inclusive com
janelas independentes. Para gerar o painel diagnóstico legado de seis gráficos,
solicite-o explicitamente:

```bash
.venv/bin/python process_dsc_data.py \
  --data data/raw/lvm/26_03_30.lvm \
  --program data/raw/pdf/26_03_30.pdf \
  --zone 2 \
  --diagnostic-panel
```

`--no-panel` continua aceito para compatibilidade com comandos existentes. O
script `analyze_step_hf.py` permanece diagnóstico; não representa, por si só,
uma medição de entalpia de transição.

Para processar vários arquivos a partir do cadastro experimental:

```bash
.venv/bin/python process_experiments_from_metadata.py \
  --metadata data/metadata/experiments.csv \
  --load-id L08 \
  --dry-run
```

Remova `--dry-run` para executar os comandos. O CSV associa cada corrida a um
experimento, carga, etapa, arquivo LVM e programa PDF.

### Análise de sinal HF por etapas

```bash
.venv/bin/python analyze_step_hf.py \
  --data-cooling data/raw/lvm/26_05_13.lvm \
  --program-cooling data/raw/pdf/26_05_13.pdf
```

Este fluxo é diagnóstico térmico/instrumental; não calcula entalpia de
transição. `dynamic_hf_deconvolution.py` oferece um fluxo separado para
deconvolução dinâmica do sinal de fluxo de calor.

### Aplicativo DSC/uDSC Calisto

Este fluxo é separado do Calvet HP: usa o programa PDF e os sinais TXT do
Calisto, e não presume que os campos ou relógios dos dois formatos sejam
equivalentes. `cli.py` permite selecionar ciclos e fornecer limites manuais
para zonas via CSV (`zone_number,start_s,end_s`). O relatório identifica a
fonte da temperatura usada e informa quando uma variável, como pressão ou
temperatura de amostra, não está disponível. O tempo do TXT é comparado ao
tempo acumulado do PDF sem correção automática; tempos não monotônicos ou
duplicados são reportados, e a seleção preserva as linhas correspondentes na
ordem original para revisão.

Para abrir a interface web:

```bash
.venv/bin/streamlit run app.py
```

Para executar a análise pela linha de comando:

```bash
.venv/bin/python cli.py \
  --pdf data/raw/pdf/23_05_12.pdf \
  --txt data/raw/lvm/23_05_12.txt \
  --max-cycles 1 \
  --outdir out/dsc_analysis_23_05_12 \
  --no-figures
```

O analisador persiste eventos, sinais, configuração, seleção de etapa e
advertências em `out/dsc_analysis/`. A detecção de eventos e a atribuição de
fase têm níveis de confiança separados. Rótulos LL/SL não são atribuídos por
padrão; `--phase-label-heuristic` apenas registra uma hipótese pendente de
revisão. Para renderizar figuras depois, sem repetir a detecção:

```bash
.venv/bin/python cli.py --outdir out/dsc_analysis_23_05_12 --plot-saved
```

Use um `--outdir` novo para cada execução/configuração; por padrão, o programa
recusa substituir resultados existentes. `--overwrite` só deve ser usado
quando a substituição for intencional.

As figuras Calisto são geradas durante a análise por padrão; use `--no-figures`
para executar apenas o processamento. Os valores automáticos são resultados
calorimétricos que precisam de revisão, não pontos de equilíbrio validados.
As curvas NIST de CO2 puro são referências de CO2 puro, inclusive quando
comparadas a experimentos de misturas.

No arquivo local `23_05_12.txt`, o parser encontrou tempo não monotônico,
valores repetidos e grandes saltos. Essas condições são gravadas como
advertências; zonas com linhas de origem descontínuas não devem ser tratadas
como alinhamento verificado. A correção manual atual usa limites de tempo; se o
relógio reinicia ou a origem do tempo não corresponde ao PDF, confirme a base
temporal antes de interpretar os eventos.

### Exportações DSC de TA Instruments TRIOS

Arquivos Excel binários `.xls` exportados pelo TRIOS são processados em um
fluxo separado. O leitor preserva as unidades originais, identifica planilhas
com colunas `Time`, `Temperature` e `Heat Flow`, e seleciona automaticamente
as rampas de resfriamento.

```bash
.venv/bin/python trios_cli.py \
  --xls "data/100724_etoh_nc16_0.851w_0.2rate.xls" \
  --outdir out/trios_100724_etoh_nc16
```

As saídas incluem gráficos do resfriamento em função da temperatura e do tempo,
os dados da rampa em `tables/cooling_01.csv`, o resumo
`trios_segments.csv`, os detalhes em `trios_details.json` e o primeiro onset
exotérmico em `trios_onsets.csv`. Esse onset é calculado no domínio do tempo
pela interseção da linha de base anterior com a tangente do primeiro flanco.

### Gráficos P-T, transições e perfis temporais

- `compare_pt_path_nist.py`: compara trajetórias experimentais P-T com
  isócoras e curva de saturação do CO2 da NIST.
- `plot_pt_multi.py`: desenha várias trajetórias P-T em um mesmo gráfico.
- `plot_cooling_comparison.py`: compara dados de resfriamento de vários
  experimentos; recebe um conjunto CSV de `data/plot_sets/`.
- `plot_phase_transition_points.py`: plota pontos experimentais e bibliográficos
  definidos nos CSVs de `data/phase_transitions/`.
- `plot_time_profile.py`: plota pressão e temperaturas de um arquivo LVM ao
  longo do tempo.

### Metadata e rastreabilidade

`data/metadata/experiments.csv` registra corridas e arquivos associados;
`data/metadata/loads.csv` registra a carga e campos experimentais que ainda
precisam ser preenchidos. Valores desconhecidos aparecem como `pending`: não
são inferidos nem bloqueiam gráficos exploratórios. A cadeia de rastreabilidade
é carga → corrida → etapa → evento. Complete massas, métodos de determinação
de CO2, composição, volume efetivo, programas, incidências e revisões apenas
com informações experimentais verificadas.

Consulte as opções de cada ferramenta com `--help`, por exemplo:

```bash
.venv/bin/python plot_time_profile.py --help
```

Esses scripts compartilham parsers e rotinas de plotagem em `util/`; embora
alguns produzam gráficos P-T ou de tempo relacionados, têm entradas e objetivos
diferentes.

## Dados e referências

- `data/metadata/experiments.csv`: cadastro das corridas e arquivos associados.
- `data/phase_transitions/`: pontos experimentais e dados bibliográficos em CSV.
- `data/plot_sets/`: seleções de experimentos para comparações gráficas.
- `data/literature/`: dados e artigos usados como referência.
- `nist/`: tabelas de propriedades de CO2 usadas nos gráficos de comparação.
- `data/README_DATA.txt`: orientação sobre os caminhos de entrada e saída.

Os dados brutos dos instrumentos devem permanecer em `data/raw/` localmente.
Não há arquivos `.lvm` ou PDFs de entrada incluídos no repositório.

## Testes

```bash
.venv/bin/python -m pytest -q
```

O CI executa os testes e uma verificação de sintaxe para Python 3.11 e 3.12.

Para graficar el proceso completo con temperaturas a la izquierda, presión a la
 derecha y heat flow en un panel inferior con el mismo tiempo:

```bash
.venv/bin/python plot_time_profile.py \
  --data data/raw/lvm/26_08_17.lvm \
  --temperature all --heat-flow --time-source device \
  --onset 25.43 --onset 10.41 --onset-direction cooling \
  --font-size 18 --legend-size 14 --line-width 1.7 \
  --out out/figures/time_profile/26_08_17_full_process.png
```

Los onsets son temperaturas de muestra suministradas por el usuario, no una
nueva detección automática. El programa interpola los tiempos y los demás
señales en cada cruce, y exporta una tabla `_onsets.csv`. Si hay más de un cruce
en la dirección seleccionada, acote la búsqueda con `--onset-time-min` y
`--onset-time-max` (en las unidades de `--time-unit`). Estos límites no recortan
el gráfico; `--time-min` y `--time-max` sí lo recortan. `--time-source device`
usa el reloj registrado para todas las señales; `sample` conserva el tiempo
por índice de muestras usado históricamente. No se desplazan las curvas.

Para cambiar los trazos, use `--temperature-style`, `--pressure-style` o
`--heat-flow-style` con `-`, `--`, `:` o `-.`. Por ejemplo,
`--pressure-style=-- --heat-flow-style=: --line-width 2.5`.
La extensión de `--out` selecciona PNG, PDF o SVG. Los textos de la figura están
en inglés y la leyenda superior se presenta en una columna.
