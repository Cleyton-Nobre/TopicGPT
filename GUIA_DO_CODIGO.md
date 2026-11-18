# Guia do código e da estrutura do repositório

Este repositório reúne o código de um estudo de **modelagem de tópicos com LLMs, sem rótulos
prévios**, aplicado a 3.027 títulos de vídeos do YouTube em português sobre remédios de
emagrecimento. A metodologia é inspirada no **TopicGPT** (geração → refinamento → atribuição):
em vez de listas de palavras-chave (como no BERTopic), o LLM produz categorias com **rótulo e
descrição** legíveis, e depois classifica cada título nessas categorias.

Este guia explica o que cada parte faz, como as partes se encaixam e como executá-las. Os detalhes
de linha de comando de cada experimento estão em [`opensource/README.md`](opensource/README.md) e
[`opensource/README-classificacao.md`](opensource/README-classificacao.md).

## 1. Visão geral do fluxo

Os títulos foram divididos em três partes, cada uma usada em uma etapa:

| Etapa | Dados | O que acontece | Onde está o código |
|---|---|---|---|
| 1. Geração | 1.800 títulos (`geracao.jsonl`) | Dois modelos (um fechado e um aberto) geram taxonomias candidatas, cada um em **duas execuções com ordens diferentes dos dados** (original e embaralhada) | `topic_generation.py`, notebook `topic_generation_two_order_experiments_v4` |
| 2. Refinamento (mesclagem) | Os tópicos gerados | Embeddings das categorias; pares com similaridade de cosseno acima de 0,5 são enviados ao LLM, que funde os redundantes | `topic_merging.py`, notebooks `merging` e `comparacao_taxonomias` |
| 3. Triagem | 627 títulos (`triagem.jsonl`) | Um LLM juiz classifica os mesmos títulos em cada uma das 4 taxonomias; compara-se a distribuição e a estabilidade para escolher uma taxonomia | `triagem.py` |
| 4. Classificação final | Gabarito humano de 291 títulos | Modelos classificam o gabarito com a taxonomia escolhida (3 estratégias de prompt) e são avaliados | `classificacao_experimento.py`, `plot_classificacao.py` |

A taxonomia escolhida tem 8 categorias (C1 a C8) mais **C0 = "nenhuma categoria se aplica"**. Os
rótulos estão em `opensource/prompts/classificacao/zero_shot.txt`.

O gabarito (291 títulos) foi construído a partir de 600 exemplos anotados de forma independente por
duas pessoas, mantendo só os casos em que ambas concordaram. Os 600 exemplos anotados completam os
3.027 títulos (1.800 + 627 + 600); apenas os 291 concordantes estão neste repositório.

## 2. Estrutura de pastas

```
TopicGPT/
├── README.md                    Resumo do projeto e links
├── GUIA_DO_CODIGO.md            Este guia
├── .gitignore
└── opensource/                  Todo o código, dados e resultados
    ├── topic_generation.py      Etapa 1: geração de tópicos (CLI; OpenAI ou vLLM)
    ├── topic_merging.py         Etapa 2: mesclagem/refinamento com modelo aberto (vLLM)
    ├── triagem.py               Etapa 3: triagem das taxonomias com um LLM juiz
    ├── classificacao_experimento.py   Etapa 4: classificação (2 modelos × 3 prompts)
    ├── plot_classificacao.py    Gráficos dos resultados da etapa 4
    ├── experiment_config.json   Parâmetros da geração (lote, semente, temperatura, ...)
    ├── docker-compose.yml       Geração local com vLLM
    ├── docker-compose.merge.yml Acréscimo ao anterior para executar a mesclagem
    ├── entrypoint.sh            Sobe o vLLM e executa a geração ou a mesclagem
    ├── .env.example             Modelo de variáveis do Docker (copie para .env)
    ├── requirements*.txt        Dependências por componente
    ├── data/
    │   ├── geracao.jsonl                 1.800 títulos da etapa 1
    │   ├── triagem.jsonl                 627 títulos da etapa 3
    │   ├── gabarito_final.jsonl          Gabarito (291): video_id, title, label
    │   ├── ground_truth_with_ids.jsonl   Mesmo gabarito, com category_id (C0 a C8)
    │   ├── taxonomia_mesclada.json       Taxonomia refinada (grupos com topic_ids, label, description)
    │   └── taxonomias/                   Saídas do modelo aberto (Ministral 3 14B)
    │       ├── 14B-topics_final_{order,shuff}.json       Tópicos gerados (ordem original e embaralhada)
    │       └── 14B-taxonomia_mesclada_{order,shuff}.json Tópicos após o refinamento
    ├── prompts/
    │   ├── prompt_mesclagem.txt          Prompt da mesclagem (versão do notebook)
    │   ├── prompt_juiz_taxonomias.txt    Prompt da avaliação qualitativa de 3 taxonomias (nenhum script o carrega)
    │   └── classificacao/                zero_shot.txt, few_shot.txt, chain_of_thought.txt
    ├── notebooks/
    │   ├── topic_generation_two_order_experiments_v4.ipynb  Geração (versão original, OpenAI)
    │   ├── merging.ipynb                                    Mesclagem (versão original)
    │   └── comparacao_taxonomias.ipynb                      Semelhança entre duas taxonomias
    ├── outputs/                      Artefatos gerados (resultados dos experimentos)
    │   ├── experiment_{1_original,2_shuffled}_order/        Geração com o modelo aberto
    │   ├── summary_all.json
    │   └── classificacao/                                   Etapa 4 (veja a seção 3.4)
    └── tests/                        Testes automatizados (unittest)
```

Os caminhos padrão dos scripts são relativos à própria pasta `opensource/`, e os notebooks usam
caminhos relativos a `opensource/notebooks/` (por isso aparecem como `../data/...`).

## 3. O que cada componente faz

### 3.1 Geração de tópicos: `topic_generation.py`

Percorre os títulos em **lotes** (100 por padrão). Em cada lote, o modelo recebe os tópicos já
existentes e os novos títulos e devolve **apenas tópicos novos** (`new_topics`), cada um com
`label`, `description` e `evidence_video_ids`. Começa com a taxonomia vazia (ou com uma taxonomia
inicial, se informada).

Pontos importantes do protocolo:

- **Duas ordens, para medir sensibilidade:** `--experiment original|shuffled|both`. A ordem
  embaralhada usa uma semente fixa (`shuffle_seed`). É essa comparação que mostrou que o resultado
  muda com a ordem dos dados.
- **Saída estruturada e validada:** o esquema JSON restringe os IDs de evidência aos do lote, e
  `validate_result` rejeita respostas inválidas; há até 3 tentativas, com o erro incluído no novo
  pedido.
- **Dois provedores, mesmo protocolo:** `--provider openai` (modelo fechado) ou `vllm` (modelo
  aberto local, Ministral 3 14B Reasoning). Só mudam o modelo, o endpoint, a credencial e a
  temperatura.
- **Auditoria:** cada chamada gera um log; há `summary.json`, `metrics.csv` (tokens, tempo,
  tentativas), `order_manifest.json` e hashes de configuração.
- Os parâmetros experimentais ficam em `experiment_config.json`.

Saída principal: `topics_final.json` (lista `topics`).

### 3.2 Refinamento: `topic_merging.py` e notebooks

Reduz redundância entre os tópicos gerados:

1. Calcula **embeddings** de `rótulo: descrição` (`paraphrase-multilingual-MiniLM-L12-v2`).
2. Seleciona pares com **similaridade de cosseno ≥ 0,5** e os ordena.
3. Envia **todos os tópicos e pares numa só análise** ao LLM, que decide os grupos. Os pares são
   só sugestões, e a similaridade não é tratada como transitiva.
4. **Valida** a resposta: cada ID original deve aparecer exatamente uma vez, sem IDs inventados
   nem grupos vazios (até 3 tentativas).

Saída: `taxonomia_mesclada.json`, com `groups` (`topic_ids`, `label`, `description`,
`justification`). `notebooks/merging.ipynb` é a versão original dessa etapa;
`notebooks/comparacao_taxonomias.ipynb` consolida tópicos próximos dentro de cada taxonomia e mede
a semelhança entre duas taxonomias (similaridade semântica bidirecional, cobertura e F1 de
cobertura). Essa pontuação é exploratória, e não uma porcentagem de acerto.

### 3.3 Triagem: `triagem.py`

Classifica os 627 títulos de `triagem.jsonl` em **cada uma das 4 taxonomias** (lotes de 10, juiz via
OpenRouter), com a opção "nenhuma categoria se aplica" (C0). O objetivo **não é medir acerto**, e sim
detectar anomalias: categoria dominante, categorias quase vazias, proporção de C0. Ao final, compara
as classificações obtidas com as diferentes taxonomias usando **NMI, ARI e P1**, métricas que
comparam partições e não dependem de os códigos terem o mesmo significado.

Cada lote concluído é salvo em disco, e uma nova execução retoma de onde parou. Com
`DRY_RUN = True` só os prompts são gerados. Os parâmetros ficam nas constantes do início do arquivo.

> **Atenção:** `taxonomia_1` e `taxonomia_2` ainda apontam para caminhos do Windows do autor (saídas
> do modelo fechado, que não estão no repositório). Ajuste `TAXONOMY_FILES` em `triagem.py` antes
> de executar. As taxonomias 3 e 4 já usam `data/taxonomias/`.

### 3.4 Classificação final: `classificacao_experimento.py`

Compara **modelos × estratégias de prompt** nos 291 títulos do gabarito, via OpenRouter. O
repositório contém os resultados de **DeepSeek V4.1 Flash** e **Gemini 3.8 Flash**.

- **Estratégias:** `zero_shot` (categorias e regras), `few_shot` (um exemplo por categoria) e `cot`
  (exemplos com um raciocínio breve antes do código). Os prompts são lidos de
  `prompts/classificacao/`.
- **Lotes de 10 títulos**, embaralhados com semente fixa; o mesmo lote é usado em todas as
  condições, então a comparação é pareada. `check_fewshot_leakage` avisa se algum exemplo do few-shot
  também estiver entre os títulos avaliados.
- **Saída estruturada** (`json_schema`, com queda para `json_object`), validação, até 3 tentativas
  e **retomada** (`run_signature.txt` impede misturar execuções incompatíveis).
- **Modos** (`--mode`): `dry_run` (só gera prompts/requisições), `run` (chama a API e avalia) e
  `evaluate` (recalcula as métricas a partir do que já está em disco, sem usar a API).
- **Métricas:** acurácia, **F1 macro**, F1 por classe, matrizes de confusão, lista de erros, custo,
  tokens e latência, além do **teste exato de McNemar** entre pares de condições.

Artefatos em `outputs/classificacao/`: `prompts/`, `requests/`, `runs/<modelo>/<prompt>/`
(`raw/`, `parsed/`, `predictions.jsonl`), `resultados/` (`metricas.csv`, `f1_por_classe.csv`,
`confusao_*.csv`, `erros.csv`, `mcnemar.csv`), `usage.csv`, `mapeamento_lotes.csv` e `figures/`.

`plot_classificacao.py` lê `resultados/` e gera, sem interface gráfica, 7 figuras (desempenho,
variação em relação ao zero-shot, custo e latência, tokens, F1 por classe, matrizes de confusão,
qualidade × custo) em PNG e PDF, com um manifesto de proveniência.

### 3.5 Docker e variáveis de ambiente

`docker-compose.yml` + `entrypoint.sh` sobem um servidor **vLLM** com o modelo de `MODEL_ID` e
executam a geração; com `docker-compose.merge.yml` executam a mesclagem (`TOPIC_TASK=merging`).
As variáveis ficam em `.env`. **Esse arquivo não é versionado:** copie `.env.example` para `.env`
e preencha `HF_TOKEN` (token do Hugging Face) localmente.

## 4. Como executar (resumo)

Os comandos partem da raiz do repositório.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r opensource/requirements.txt

# Testes
cd opensource && PYTHONPATH=. python3 -m unittest discover -s tests

# Reavaliar a classificação já feita (sem API)
python3 classificacao_experimento.py --mode evaluate

# Gerar os gráficos
pip install -r requirements-plots.txt
python3 plot_classificacao.py

# Nova execução com a API (precisa de OPENROUTER_API_KEY)
python3 classificacao_experimento.py --mode run --output-dir outputs/classificacao_nova
```

Geração e mesclagem com o modelo aberto, via Docker, estão em `opensource/README.md`.

## 5. Formatos de dados

- **Títulos** (`geracao.jsonl`, `triagem.jsonl`): JSONL com `video_id` e `title` (e metadados do vídeo).
- **Gabarito** (`ground_truth_with_ids.jsonl`): `video_id`, `title`, `label`, `category_id` (C0 a C8).
- **Tópicos** (`topics_final.json`): objeto com `topics` (`label`, `description`, `evidence_video_ids`).
- **Taxonomia refinada** (`taxonomia_mesclada.json`): objeto com `groups` (`topic_ids`, `label`, `description`, `justification`).

## 6. Cuidados

- **Segredos:** nunca versione o `.env`. Um `HF_TOKEN` esteve versionado no histórico antigo deste
  repositório e deve ser **revogado** em huggingface.co/settings/tokens.
- **Reprodutibilidade:** a geração depende da ordem dos dados (por isso as duas ordens), e as
  condições da classificação têm uma execução cada; diferenças pequenas entre condições não são
  conclusivas sem repetições ou testes de significância.
- **Custos:** o custo do modelo aberto aparece como zero nos CSVs porque não inclui a infraestrutura.
