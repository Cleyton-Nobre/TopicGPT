# Experimento de classificação

O script `classificacao_experimento.py` compara dois modelos via OpenRouter
com três variações de prompt: `zero_shot`, `few_shot` e `cot`. O conjunto importado
contém 291 títulos com gabarito nas categorias C0 a C8. Os identificadores dos
modelos e os parâmetros foram preservados conforme o experimento original.

## Organização dos arquivos

Os arquivos de `experiment` foram incorporados seguindo a estrutura existente:
código executável em `opensource`, entradas em `data` e artefatos gerados em `outputs`.
Os templates de classificação têm uma pasta própria para distingui-los dos prompts
gerados por lote. Os caminhos abaixo são relativos a `opensource`.

| Origem em `experiment/` | Destino | Conteúdo |
| --- | --- | --- |
| `classificacao_experimento (1).py` | `classificacao_experimento.py` | Script, com nome normalizado e caminhos ajustados. |
| `ground_truth_with_ids.jsonl` | `data/ground_truth_with_ids.jsonl` | Títulos, IDs e gabarito. |
| `zero_shot.txt`, `few_shot.txt`, `chain_of_thought.txt` | `prompts/classificacao/` | Três templates originais. |
| `experimento/prompts/` | `outputs/classificacao/prompts/` | 30 prompts renderizados por estratégia. |
| `experimento/requests/` | `outputs/classificacao/requests/` | Requisições de exemplo do primeiro lote, uma por condição. |
| `experimento/mapeamento_lotes.csv` | `outputs/classificacao/mapeamento_lotes.csv` | Correspondência entre lote, posição e ID do vídeo. |
| `experimento/usage.csv` | `outputs/classificacao/usage.csv` | Tokens, custos, latências e tentativas. |
| `experimento/runs/` | `outputs/classificacao/runs/` | Execuções por modelo e estratégia, incluindo `raw/`, `parsed/`, `predictions.jsonl` e `run_signature.txt`. |
| `experimento/resultados/` | `outputs/classificacao/resultados/` | Métricas, F1 por classe, matrizes de confusão, erros e McNemar. |
| `.venv/` | Não incorporado | Ambiente local; recrie-o usando `requirements.txt`. |

Os dados, templates e artefatos importados mantêm seu conteúdo original. As assinaturas
de execução também foram preservadas para permitir a retomada. A pasta `experiment`
original foi mantida como cópia de origem; ela não é necessária para executar o script integrado.

## Execução

A partir da raiz de `TopicGPT`, prepare as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r opensource/requirements.txt
```

Para conferir prompts e requisições sem chamar a API, use uma saída temporária:

```bash
python3 opensource/classificacao_experimento.py --mode dry_run \
  --output-dir /tmp/topicgpt-classificacao-preview
```

Para recalcular as métricas das execuções importadas, sem chamar a API:

```bash
python3 opensource/classificacao_experimento.py --mode evaluate
```

Esse comando regrava os CSVs de `resultados/` e os arquivos `predictions.jsonl`
a partir das respostas em `runs/<modelo>/<prompt>/parsed/`.

Para executar ou retomar chamadas, defina `OPENROUTER_API_KEY` no ambiente e use:

```bash
python3 opensource/classificacao_experimento.py --mode run
```

O modo padrão continua sendo `run`, como no script original. As seis condições
importadas têm 30 lotes concluídos cada; com as configurações preservadas, o script
aproveita os arquivos `parsed/` existentes. Para uma nova execução, escolha outro
`--output-dir`. Mudanças incompatíveis com uma execução existente são rejeitadas
pela verificação de `run_signature.txt`.

Os caminhos padrão são resolvidos a partir da localização do script. Também é
possível executá-lo de outro diretório usando seu caminho absoluto. `--input`
seleciona outro JSONL e `--output-dir` seleciona outro diretório de artefatos;
caminhos relativos fornecidos nessas opções partem do diretório atual.
Os demais parâmetros continuam nas constantes do script.

## Gráficos de comparação

O script [`plot_classificacao.py`](plot_classificacao.py) lê os CSVs de
`outputs/classificacao/resultados/` e grava as figuras em
`outputs/classificacao/figures/`. Ele funciona sem interface gráfica, sem credenciais
e sem chamar os modelos. As dependências de visualização são separadas das dependências
de inferência. A partir da raiz de `TopicGPT`, com o ambiente virtual ativado:

```bash
python3 -m pip install -r opensource/requirements-plots.txt
python3 opensource/plot_classificacao.py
```

São geradas sete figuras, cada uma em **PNG a 300 dpi e PDF vetorial**:

| Arquivo, sem extensão | Comparação |
| --- | --- |
| `01_desempenho` | Acurácia e F1 macro por modelo e prompt. |
| `02_variacao_vs_zero_shot` | Diferença em pontos percentuais contra o zero-shot do próprio modelo. |
| `03_custo_latencia` | Custo total registrado em dólares e latência média por chamada. |
| `04_tokens` | Totais de tokens de entrada e saída, incluindo tentativas registradas. |
| `05_f1_por_classe` | F1 das seis condições por categoria, com o suporte de cada classe. |
| `06_matrizes_confusao` | As seis matrizes; linhas reais, colunas preditas, contagens e percentuais por linha. |
| `07_qualidade_custo` | Relação entre custo total e F1 macro. |

Azul identifica DeepSeek e laranja identifica Gemini nos gráficos comparativos.
As escalas de F1 e das matrizes de confusão são compartilhadas entre condições.
O gráfico de dispersão de qualidade e custo amplia o eixo de F1 para distinguir os
pontos próximos; os gráficos de barras de desempenho começam em zero.
O script exige as seis condições completas, com totais e suportes iguais, e valida
as métricas contra as matrizes antes de salvar as figuras. Entradas inválidas
produzem erro, em vez de uma comparação parcial.

Para selecionar outra execução e exportar somente PNG e SVG:

```bash
python3 opensource/plot_classificacao.py \
  --results-dir opensource/outputs/outra_execucao/resultados \
  --output-dir opensource/outputs/outra_execucao/figures \
  --formats png svg --dpi 300
```

Os caminhos padrão independem do diretório atual. Caminhos relativos informados na
CLI partem do diretório atual. Ao repetir o comando, as figuras com os mesmos nomes
e o `manifest.json` são substituídos; os CSVs de avaliação são preservados.

O manifesto registra hashes SHA-256 dos oito CSVs usados, do script e das figuras,
além das versões de Python, Matplotlib e NumPy, formatos, resolução e cobertura
de cada condição. Para reproduzir uma versão, preserve esses arquivos e as versões
registradas no manifesto.

As diferenças exibidas são descritivas de uma execução, sem intervalos de confiança.
Elas não indicam significância estatística; o arquivo `mcnemar.csv` do experimento
é uma análise separada e não é usado nessas figuras. Custos vêm dos totais registrados,
latência é a média por chamada e tokens dependem do tokenizer de cada modelo.
Não se soma novamente o raciocínio ao total de tokens de saída.

Para validar a leitura dos resultados e a exportação nos três formatos:

```bash
cd opensource
python3 -m unittest discover -s tests -p 'test_plot_classificacao.py' -v
```
