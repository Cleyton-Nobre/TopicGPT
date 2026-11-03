# Comparação de LLMs na geração de tópicos

A CLI executa o processamento em `topic_generation.py` para ambos os provedores.
O provedor OpenAI seleciona `gpt-6.1-sol`; o Docker seleciona o modelo local em `MODEL_ID`.
As chamadas usam Chat Completions e diferem no identificador do modelo,
endpoint, credencial e temperatura. O processamento, os prompts, o esquema JSON com IDs restritos
ao lote, a validação, as tentativas e o formato dos resultados são compartilhados.

O notebook atualmente na raiz contém uma implementação independente mais antiga.
As configurações e regras descritas abaixo se aplicam à CLI e ao Docker.

Os parâmetros experimentais ficam em **`experiment_config.json`**:

| Parâmetro | Padrão |
| --- | --- |
| Títulos por lote | 100 |
| Tentativas por resposta inválida | 3 |
| Semente para embaralhar títulos | 20261026 |
| Limite de tokens de geração | 2048, incluindo raciocínio e resposta |
| Temperatura do vLLM (`vllm_temperature`) | `0.0`, como na execução local anterior |
| Esforço de raciocínio | `high`, compatível com o Mistral local e o GPT-6.1 Sol |
| Timeout da chamada | 600 segundos |
| Retentativas de transporte do SDK | 2 |

Altere esse arquivo para os dois provedores. As antigas variáveis `BATCH_SIZE`,
`MAX_ATTEMPTS`, `SHUFFLE_SEED`, `MAX_TOKENS` e `TEMPERATURE` do `.env` foram
substituídas por essa configuração. A CLI permite sobrescrever parâmetros
explicitamente; os hashes registram essas mudanças.

Para ajustar a temperatura local, altere `vllm_temperature` em
`experiment_config.json` (valor entre 0 e 2). A CLI também aceita
`--provider vllm --temperature 0.0`, que prevalece sobre o JSON. Essa configuração
é enviada somente ao vLLM e aparece no terminal, no protocolo e em
`generation_parameters` de `topics_final.json` e `summary.json`.

Instale `requirements.txt` e execute a CLI dentro de `opensource`, com
`OPENAI_API_KEY` definida no ambiente para usar o provedor OpenAI:

```bash
python3 topic_generation.py --provider openai --model gpt-6.1-sol
```

Para o modelo local, mantenha `MODEL_ID`, `SERVED_MODEL_NAME` e as configurações de
hardware no `.env`, e execute:

```bash
docker compose up --abort-on-container-exit --exit-code-from topic-job
```

O modelo configurado, `mistralai/Ministral-3-14B-Reasoning-2512`, recebe automaticamente
o parser de raciocínio `mistral` e os formatos de tokenizer, configuração e pesos
correspondentes. Para outro modelo, configure `REASONING_PARSER`, `TOKENIZER_MODE`,
`CONFIG_FORMAT` e `LOAD_FORMAT` conforme sua documentação. O servidor usa
`--generation-config vllm` para evitar parâmetros de amostragem escondidos na
configuração dos pesos. Veja a [receita oficial do Ministral](https://docs.vllm.ai/projects/recipes/en/stable/Mistral/Ministral-3-Reasoning.html).

Para Mistral, o servidor usa `--attention-backend TRITON_ATTN` para evitar o
[problema de saída corrompida com FlashInfer em GPUs Blackwell, reproduzido na RTX 5090](https://github.com/vllm-project/vllm/issues/53481).
O backend pode ser sobrescrito por `ATTENTION_BACKEND` no `.env`.

Cada execução processa todos os títulos e **sobrescreve as saídas dos experimentos selecionados**
diretamente no destino configurado, sem criar pastas `run_`. A CLI OpenAI grava em
`outputs/openai`; o Docker grava em `outputs` conforme o Compose. Antes de processar,
o gerador remove as pastas dos experimentos selecionados, `protocol.json` e
`summary_all.json`. Outros arquivos e a saída da outra LLM são preservados. Pastas
`run_` criadas pela versão anterior não são usadas nem removidas automaticamente.
As pastas das ordens original e
embaralhada contêm `topics_final.json`, `summary.json`, `metrics.csv`, os resultados
por lote em `batches/` e logs das tentativas em `logs/`. Os arquivos por lote são
para auditoria; não há retomada por checkpoint em nenhum dos provedores.

Cada experimento pode receber uma taxonomia inicial independente, no formato do
**`topics_final.json` gerado por este programa** (objeto com a lista `topics`).
Informe os caminhos pela CLI:

```bash
python3 topic_generation.py --provider openai \
  --initial-taxonomy-original data/topics_final_order.json \
  --initial-taxonomy-shuffled data/topics_final_shuff.json
```

Cada ordem tem duas possibilidades: **começar vazia** ou **começar com os K tópicos
do JSON informado**. K é a quantidade de itens na lista `topics` desse arquivo.
Sem entrada, o padrão é começar vazio.

Na CLI e no Docker, os caminhos também podem ser definidos pelas variáveis `INITIAL_TAXONOMY_ORIGINAL`
e `INITIAL_TAXONOMY_SHUFFLED`. Para o Docker começar vazio, deixe no `.env`:

```dotenv
INITIAL_TAXONOMY_ORIGINAL=
INITIAL_TAXONOMY_SHUFFLED=
```

Para o Docker importar os arquivos, configure os caminhos vistos pelo contêiner:

```dotenv
INITIAL_TAXONOMY_ORIGINAL=/app/data/topics_final_order.json
INITIAL_TAXONOMY_SHUFFLED=/app/data/topics_final_shuff.json
```

As escolhas são independentes: uma ordem pode começar vazia e a outra com K tópicos.
É possível fornecer o mesmo JSON para ambas. Os tópicos importados entram
no primeiro prompt e são preservados integralmente; os novos IDs continuam após
o maior `topic_id` importado, inclusive quando há lacunas. Todos os títulos são
processados novamente. Evidências e `created_in_batch` importados conservam seu
significado na execução de origem; as novas evidências devem pertencer ao lote atual.

O carregamento verifica os campos de cada tópico, IDs e rótulos duplicados antes
de limpar qualquer saída ou chamar a API. Somente as taxonomias das ordens
selecionadas por `--experiment` são carregadas. É possível usar como entrada uma
saída que será sobrescrita: os arquivos são lidos antes da limpeza, e cada
experimento salva os tópicos iniciais em `initial_taxonomy.json` para auditoria.
`topics_final.json` e `summary.json` registram `initial_topics`, `generated_topics`,
`initial_taxonomy_file` e `initial_taxonomy_hash`; `total_topics` inclui os importados.

Quando a taxonomia está vazia, o primeiro lote precisa gerar pelo menos um tópico.
Essa regra é aplicada no prompt, no JSON Schema e no validador. Com tópicos já
importados ou gerados, `new_topics: []` é aceito quando não houver temas novos.
O leitor aceita JSON puro ou envolvido por crases Markdown, inclusive sem as crases
de fechamento, e preserva a resposta original no log. Texto adicional, JSON
incompleto e evidências inválidas continuam sendo rejeitados.

Compare sempre a mesma ordem e as mesmas taxonomias iniciais entre os modelos.
O **`comparison_hash`** inclui o código, o protocolo, os parâmetros, os registros
na ordem processada e os tópicos da taxonomia inicial, sem incluir a identidade
da LLM ou o caminho do JSON importado. A temperatura explícita somente no vLLM
faz os hashes dos dois provedores diferirem. Confira `input_hash` e
`initial_taxonomy_hash` para comparar as entradas e consulte `protocol.json`
para verificar os parâmetros de cada execução. O histórico de tópicos diverge conforme as
respostas de cada modelo; isso faz parte do experimento incremental. Tópicos com
rótulos duplicados ou evidências inválidas são rejeitados e seguem a mesma regra de
nova tentativa nos dois provedores. Não há remoção automática dessas duplicações.

O GPT-6.1 Sol exige raciocínio e não aceita `temperature` ou `top_p` nesse modo.
Por isso, a execução OpenAI omite esses controles; o vLLM recebe a temperatura
local configurada. Ambos recebem o mesmo `reasoning_effort`. Veja a
[documentação oficial da OpenAI](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.1-sol).
O tokenizer Mistral aceita somente `none` e `high`; o padrão compartilhado é `high`
para manter raciocínio habilitado nos dois modelos. Uma configuração incompatível
com o Mistral é rejeitada antes de limpar saídas ou chamar o servidor.
O significado de `high`, a amostragem interna, o tokenizer e o mecanismo de saída
estruturada dependem da LLM e do servidor. Temperatura zero também não garante
resultados idênticos entre execuções. A semente controla somente a ordem dos títulos.

As métricas de tempo e tokens usam o mesmo formato e os totais incluem tentativas
inválidas que precederam um lote válido. Os logs também preservam tentativas de um
lote que esgotou seu limite. Tokens de raciocínio e cache são registrados quando
informados; campo vazio significa desconhecido. O tempo inclui a chamada e a
validação, e a taxa usa todos os tokens de conclusão informados pelo servidor.
Latência e tokens dependem também da infraestrutura e do tokenizer. Não há
estimativa monetária automática. Se ocorrer truncamento, ajuste
`max_completion_tokens` para ambos e confira `MAX_MODEL_LEN` do servidor local:
ele precisa comportar o prompt crescente mais a geração.

Para verificar o processamento com respostas simuladas, sem consultar as LLMs:

```bash
python3 -m unittest discover -s tests -v
```
