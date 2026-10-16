Os arquivos brutos dos instrumentos devem ser colocados nestas pastas locais:

- Arquivos `.lvm`: `data/raw/lvm/`
- PDFs dos programas térmicos: `data/raw/pdf/`

Figuras, tabelas e relatórios gerados são gravados em `out/`.

O cadastro `metadata/experiments.csv` relaciona corridas aos arquivos de
entrada; `metadata/loads.csv` mantém os dados de cada carga. Campos
experimentais ainda não informados estão marcados como `pending` e devem ser
preenchidos apenas a partir de registros verificados. Preserve a rastreabilidade
carga → corrida → etapa → evento e não altere os arquivos brutos.

Os formatos Calvet HP (`.lvm` + PDF) e Calisto (`.txt` + PDF) são processados
por fluxos separados. Os nomes de sinais, unidades, temperaturas e bases
temporais registrados nos resultados dependem do formato e da fonte disponível.
Não aplique correções de sincronização ou de atraso térmico sem evidência
experimental independente.

Os scripts ainda aceitam arquivos diretamente em `data/` por compatibilidade
com a organização antiga, mas `data/raw/` é o local recomendado. Os arquivos
brutos não são versionados pelo Git.
