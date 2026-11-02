# Comparação de LLMs na geração de tópicos

O notebook e a CLI executam o mesmo processamento em `topic_generation.py`.
O notebook seleciona `gpt-6.1-sol`; o Docker seleciona o modelo local em `MODEL_ID`.
As chamadas usam Chat Completions e diferem apenas no identificador do modelo,
endpoint e credencial. O processamento, os prompts, o esquema JSON com IDs restritos
ao lote, a validação, as tentativas e o formato dos resultados são compartilhados.

Os parâmetros experimentais ficam em **`experiment_config.json`**:

| Parâmetro | Padrão |
| --- | --- |
| Títulos por lote | 100 |
| Tentativas por resposta inválida | 3 |
| Semente para embaralhar títulos | 20261026 |
| Limite de tokens de geração | 2048, incluindo raciocínio e resposta |
| Esforço de raciocínio | `high`, compatível com o Mistral local e o GPT-6.1 Sol |
| Timeout da chamada | 600 segundos |
| Retentativas de transporte do SDK | 2 |

Altere esse arquivo para os dois provedores. As antigas variáveis `BATCH_SIZE`,
`MAX_ATTEMPTS`, `SHUFFLE_SEED`, `MAX_TOKENS` e `TEMPERATURE` do `.env` foram
substituídas por essa configuração. A CLI permite sobrescrever parâmetros
explicitamente; os hashes registram essas mudanças.

Instale `requirements.txt` no ambiente do notebook e execute suas células a partir
da pasta do projeto, com `OPENAI_API_KEY` definida no ambiente. Mantenha o notebook,
o `.py` e o JSON de configuração juntos. A entrada padrão é `data/geracao.jsonl`.
A mesma execução OpenAI está disponível pela CLI:

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

Cada execução começa do zero e **sobrescreve as saídas dos experimentos selecionados**
diretamente no destino configurado, sem criar pastas `run_`. O notebook grava em
`outputs/openai`; o Docker grava em `outputs` conforme o Compose. Antes de processar,
o gerador remove as pastas dos experimentos selecionados, `protocol.json` e
`summary_all.json`. Outros arquivos e a saída da outra LLM são preservados. Pastas
`run_` criadas pela versão anterior não são usadas nem removidas automaticamente.
As pastas das ordens original e
embaralhada contêm `topics_final.json`, `summary.json`, `metrics.csv`, os resultados
por lote em `batches/` e logs das tentativas em `logs/`. Os arquivos por lote são
para auditoria; não há retomada por checkpoint em nenhum dos provedores.

Compare sempre a mesma ordem entre os modelos. O **`comparison_hash`** nos respectivos
`summary.json` deve coincidir: ele inclui o código, o protocolo, os parâmetros e os
registros na ordem processada, sem incluir a identidade da LLM. `protocol.json`
permite conferir as condições completas. O histórico de tópicos diverge conforme as
respostas de cada modelo; isso faz parte do experimento incremental. Tópicos com
rótulos duplicados ou evidências inválidas são rejeitados e seguem a mesma regra de
nova tentativa nos dois provedores. Não há remoção automática dessas duplicações.

O GPT-6.1 Sol exige raciocínio e não aceita `temperature` ou `top_p` nesse modo.
Por isso, os dois provedores recebem o mesmo `reasoning_effort` e omitem esses
controles. Veja a [documentação oficial da OpenAI](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.1-sol).
O tokenizer Mistral aceita somente `none` e `high`; o padrão compartilhado é `high`
para manter raciocínio habilitado nos dois modelos. Uma configuração incompatível
com o Mistral é rejeitada antes de limpar saídas ou chamar o servidor.
Isso iguala o procedimento e os parâmetros enviados; o significado de `high`, a
amostragem interna, o tokenizer e o mecanismo de saída estruturada dependem da LLM
e do servidor. O mesmo hash não garante orçamento interno de raciocínio igual nem
resultados determinísticos. A semente controla somente a ordem dos títulos.

As métricas de tempo e tokens usam o mesmo formato e os totais incluem tentativas
inválidas que precederam um lote válido. Os logs também preservam tentativas de um
lote que esgotou seu limite. Tokens de raciocínio e cache são registrados quando
informados; campo vazio significa desconhecido. O tempo inclui a chamada e a
validação, e a taxa usa todos os tokens de conclusão informados pelo servidor.
Latência e tokens dependem também da infraestrutura e do tokenizer. Não há
estimativa monetária automática. Se ocorrer truncamento, ajuste
`max_completion_tokens` para ambos e confira `MAX_MODEL_LEN` do servidor local:
ele precisa comportar o prompt crescente mais a geração.

Para verificar a equivalência com respostas simuladas, sem consultar as LLMs:

```bash
python3 -m unittest discover -s tests -v
```
