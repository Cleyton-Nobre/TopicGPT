#!/usr/bin/env bash
set -Eeuo pipefail

VLLM_PID=""
GENERATOR_PID=""

echo "[job] Ajustando permissões de arquivos para leitura e escrita" >&2
chmod -R 777 /root/.cache
chmod -R 777 /app

shutdown() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$GENERATOR_PID" ]]; then
        kill -TERM "$GENERATOR_PID" 2>/dev/null || true
        wait "$GENERATOR_PID" 2>/dev/null || true
    fi
    if [[ -n "$VLLM_PID" ]]; then
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

: "${MODEL_ID:=mistralai/Ministral-3-14B-Reasoning-2512}"
: "${SERVED_MODEL_NAME:=topic-local}"
: "${VLLM_API_KEY:=local-token}"
: "${DTYPE:=auto}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.85}"
: "${TENSOR_PARALLEL_SIZE:=1}"

# A separação entre raciocínio e JSON final depende do parser do modelo local.
case "$MODEL_ID" in
    mistralai/Ministral-3-*-Reasoning*|mistralai/Magistral*)
        : "${REASONING_PARSER:=mistral}"
        : "${TOKENIZER_MODE:=mistral}"
        : "${CONFIG_FORMAT:=mistral}"
        : "${LOAD_FORMAT:=mistral}"
        ;;
esac
if [[ -z "${REASONING_PARSER:-}" ]]; then
    echo "[job] Defina REASONING_PARSER para o modelo de raciocínio em MODEL_ID." >&2
    exit 1
fi
: "${TOKENIZER_MODE:=auto}"
: "${CONFIG_FORMAT:=auto}"
: "${LOAD_FORMAT:=auto}"
# Evita defaults de amostragem ocultos no generation_config.json dos pesos.
GENERATION_CONFIG=vllm
export MODEL_ID REASONING_PARSER TOKENIZER_MODE CONFIG_FORMAT LOAD_FORMAT GENERATION_CONFIG
export MAX_MODEL_LEN DTYPE TENSOR_PARALLEL_SIZE

# O vLLM inicia em background, mas a API só fica acessível depois de carregar os pesos.
server_args=(
    serve "$MODEL_ID"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port 8000
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --api-key "$VLLM_API_KEY"
    --reasoning-parser "$REASONING_PARSER"
    --tokenizer-mode "$TOKENIZER_MODE"
    --config-format "$CONFIG_FORMAT"
    --load-format "$LOAD_FORMAT"
    --generation-config "$GENERATION_CONFIG"
)
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
    server_args+=(--enforce-eager)
fi

echo "[job] Iniciando vLLM com modelo ${MODEL_ID} (alias ${SERVED_MODEL_NAME})" >&2
vllm "${server_args[@]}" &
VLLM_PID=$!

# O script aguarda /v1/models identificar o modelo antes de gerar os tópicos.
echo "[job] Iniciando geração dentro do mesmo contêiner" >&2
python3 -u /app/topic_generation.py --provider vllm &
GENERATOR_PID=$!

# Se o servidor falhar durante o processamento, finalize o job com erro,
# em vez de deixar o gerador aguardando indefinidamente.
while kill -0 "$GENERATOR_PID" 2>/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[job] ERRO: o processo vLLM terminou antes do gerador." >&2
        exit 1
    fi
    sleep 2
done

status=0
wait "$GENERATOR_PID" || status=$?
if [[ "$status" -eq 0 ]]; then
    echo "[job] Geração finalizada. Parando vLLM e liberando GPU." >&2
else
    echo "[job] Gerador terminou com código $status. Confira os logs e outputs." >&2
fi
exit "$status"
