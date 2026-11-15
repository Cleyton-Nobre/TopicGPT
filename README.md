# TopicGPT

O experimento de classificação via OpenRouter está integrado em
[`opensource/classificacao_experimento.py`](opensource/classificacao_experimento.py).
O gabarito fica em `opensource/data`, os templates em `opensource/prompts/classificacao`
e os prompts gerados, requisições, execuções e resultados em `opensource/outputs/classificacao`.
Consulte a [organização dos arquivos e os comandos de execução](opensource/README-classificacao.md).
Para comparar Gemini e DeepSeek com os três tipos de prompt, use
[`opensource/plot_classificacao.py`](opensource/plot_classificacao.py).
As figuras são salvas em `opensource/outputs/classificacao/figures`; as instruções
estão na [documentação dos gráficos](opensource/README-classificacao.md#gráficos-de-comparação).

Para executar a consolidação global de `merging.ipynb` usando o modelo aberto
servido pelo vLLM, use [`opensource/topic_merging.py`](opensource/topic_merging.py).
Os comandos da CLI e do Docker estão em [`opensource/README.md`](opensource/README.md).

A CLI em [`opensource`](opensource/README.md) aceita uma taxonomia
inicial para cada experimento, em ordem original e embaralhada, usando os arquivos
`topics_final.json` gerados pelo programa. Na CLI, use
`--initial-taxonomy-original` e `--initial-taxonomy-shuffled`.

Por padrão, os experimentos começam vazios. Para começar com K tópicos, informe
um JSON com K itens em `topics`. Os arquivos disponíveis são
`opensource/data/topics_final_order.json` e `opensource/data/topics_final_shuff.json`.

Com a taxonomia vazia, o primeiro lote deve gerar pelo menos um tópico. O notebook
na raiz contém uma versão independente do processamento.

Para quantificar a semelhança entre as duas abordagens, use o notebook separado
[`comparacao_taxonomias.ipynb`](comparacao_taxonomias.ipynb). Os caminhos de entrada
ficam na primeira célula de configuração; o padrão compara `topics_final_order.json`
com `topics_final_shuff.json`. Primeiro, ele mescla tópicos semanticamente próximos
dentro de cada arquivo, de forma independente, usando ligação completa e limiar
interno de `0.95`. Todos os pares de membros de um grupo precisam atingir esse limiar.
O notebook preserva os IDs e textos de origem; exibe o rótulo do tópico mais central
e representa cada grupo pela média normalizada dos embeddings de seus membros.

Depois, compara as taxonomias consolidadas com similaridade semântica bidirecional,
cobertura e F1 de cobertura, usando limiar externo de `0.5`. Também mostra as métricas
antes do merge e a redução, coesão e cobertura interna de cada abordagem. A cobertura
interna refere-se à representação dos tópicos originais, não à cobertura de todos os vídeos.
As configurações são ajustáveis na primeira célula do notebook.

O processamento usa embeddings locais e exporta os grupos consolidados, suas origens,
as métricas, a matriz e as correspondências para
`outputs/comparacao_taxonomias_consolidadas`. A pontuação é exploratória e não representa
porcentagem de acerto. As dependências estão em
[`requirements-comparison.txt`](opensource/requirements-comparison.txt).
