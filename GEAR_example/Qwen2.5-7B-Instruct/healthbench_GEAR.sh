#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================================
# GEAR DAG HealthBench GRPO training
#
# Called by the outer train.sh script on the Ray head node.
# Node roles:
#   worker0      = grader/vLLM service
#   worker1-7    = RL training
#   RL nodes     = 7
#   NPUs per node = 8
# ============================================================

# ============================================================
# Easy knobs: test the reward path first
#
# DEBUG_REWARD_RUN=1 by default:
#   - use a small batch
#   - run only one step
#   - disable multi-response judging to avoid requesting 100+ decisions at once
#   - enable raw judge response debugging
#   - stop immediately on parse failures instead of silently returning reward=0
#
# For a full training run after reward validation:
#   export DEBUG_REWARD_RUN=0
#   export GEAR_DEBUG_JUDGE=0
#   export GEAR_FAIL_ON_PARSE_ERROR=1
#   export TRAIN_BATCH_SIZE=128
#   export PPO_MINI_BATCH_SIZE=64
#   export ROLLOUT_N=8
#   export TOTAL_TRAINING_STEPS=350
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd -P)}"

DEBUG_REWARD_RUN="${DEBUG_REWARD_RUN:-1}"

# ------------------------------
# Short reward debugging run
# Export DEBUG_REWARD_RUN=1 before submission when the reward path needs investigation.
# ------------------------------
DEBUG_TRAIN_BATCH_SIZE="${DEBUG_TRAIN_BATCH_SIZE:-56}"
DEBUG_PPO_MINI_BATCH_SIZE="${DEBUG_PPO_MINI_BATCH_SIZE:-56}"
DEBUG_ROLLOUT_N="${DEBUG_ROLLOUT_N:-1}"
DEBUG_TOTAL_TRAINING_STEPS="${DEBUG_TOTAL_TRAINING_STEPS:-1}"
DEBUG_TOTAL_EPOCHS="${DEBUG_TOTAL_EPOCHS:-1}"
DEBUG_TEST_FREQ="${DEBUG_TEST_FREQ:-999999}"
DEBUG_VAL_BEFORE_TRAIN="${DEBUG_VAL_BEFORE_TRAIN:-False}"
DEBUG_RESUME_MODE="${DEBUG_RESUME_MODE:-disable}"
DEBUG_SAVE_FREQ="${DEBUG_SAVE_FREQ:-999999}"

DEBUG_GEAR_GROUP_RESPONSES_PER_PROMPT="${DEBUG_GEAR_GROUP_RESPONSES_PER_PROMPT:-0}"
DEBUG_GEAR_MAX_RESPONSES_PER_JUDGE="${DEBUG_GEAR_MAX_RESPONSES_PER_JUDGE:-1}"
DEBUG_GEAR_DEBUG_JUDGE="${DEBUG_GEAR_DEBUG_JUDGE:-1}"

# ------------------------------
# Default full training configuration: start with a small batch
# Seven nodes with eight NPUs each provide 56 NPUs, so TRAIN_BATCH_SIZE defaults to 56.
# ------------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-56}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-56}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
RESUME_MODE="${RESUME_MODE:-disable}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

# ------------------------------
# Common performance settings
# ------------------------------
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"

# Start conservatively and increase to 2 or 4 after the run is stable.
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}"

TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"

# ------------------------------
# Common reward and grader settings
# ------------------------------
GEAR_MODE="${GEAR_MODE:-dag}"
#PI_MODE="${PI_MODE:-binary}"
PI_MODE="${PI_MODE:-judge_prob}"

# Keep multi-response judging disabled initially for stability.
# Set this to 1 later if higher judge throughput is needed.
GEAR_GROUP_RESPONSES_PER_PROMPT="${GEAR_GROUP_RESPONSES_PER_PROMPT:-0}"
GEAR_MAX_RESPONSES_PER_JUDGE="${GEAR_MAX_RESPONSES_PER_JUDGE:-1}"

# Limit the number of rubrics per judge call to avoid missing keys for long rubrics.
GEAR_MAX_RUBRICS_PER_JUDGE="${GEAR_MAX_RUBRICS_PER_JUDGE:-4}"

GEAR_MAX_CONCURRENT_WORKERS="${GEAR_MAX_CONCURRENT_WORKERS:-32}"

# After disabling thinking for the Qwen3 judge, 1024 is usually sufficient. Increase to 2048 if needed.
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-1024}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-300}"
VLLM_LOAD_REFRESH_INTERVAL_SEC="${VLLM_LOAD_REFRESH_INTERVAL_SEC:-2.0}"

# Debug output.
# Raw debugging is disabled for full training runs to avoid extra files and overhead.
GEAR_DEBUG_JUDGE="${GEAR_DEBUG_JUDGE:-0}"
GEAR_FAIL_ON_PARSE_ERROR="${GEAR_FAIL_ON_PARSE_ERROR:-1}"
GEAR_DEBUG_MAX_FILES="${GEAR_DEBUG_MAX_FILES:-2000}"
GEAR_DEBUG_RAW_RESPONSE_CHARS="${GEAR_DEBUG_RAW_RESPONSE_CHARS:-12000}"
GEAR_DEBUG_ROLLOUT_CHARS="${GEAR_DEBUG_ROLLOUT_CHARS:-4000}"

# ------------------------------
# Model configuration
# ------------------------------
MODEL_PATH="${MODEL_PATH:-${RL_MODEL_PATH:-}}"

# ------------------------------
# GEAR HealthBench data with rubric DAG annotations
# ------------------------------
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/data/health_bench/healthbench_train_gear.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/data/health_bench/healthbench_val_gear.parquet}"

# ------------------------------
# Experiment configuration
# ------------------------------
EXPERIMENT_NAME="${GEAR_DAG_EXPERIMENT_NAME:-Qwen2.5-7B-Instruct_healthbench_GEAR_DAG}"
CHECKPOINT_DIR="${GEAR_DAG_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/verl_grpo_general/${EXPERIMENT_NAME}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/log/rollout_log/${EXPERIMENT_NAME}}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${PROJECT_ROOT}/log/validation_log/${EXPERIMENT_NAME}}"
GEAR_DEBUG_DIR="${GEAR_DEBUG_DIR:-${PROJECT_ROOT}/log/reward_debug/${EXPERIMENT_NAME}}"

# Seven RL nodes. worker0 hosts the grader; worker1 through worker7 run RL training.
TRAIN_NNODES="${TRAIN_NNODES:-${RL_NNODES:-7}}"

# ------------------------------
# Apply debug reward run overrides before sanity checks
# ------------------------------
if [[ "${DEBUG_REWARD_RUN}" == "1" ]]; then
    echo "DEBUG_REWARD_RUN=1: overriding batch/step/reward settings for reward debugging"

    TRAIN_BATCH_SIZE="${DEBUG_TRAIN_BATCH_SIZE}"
    PPO_MINI_BATCH_SIZE="${DEBUG_PPO_MINI_BATCH_SIZE}"
    ROLLOUT_N="${DEBUG_ROLLOUT_N}"
    TOTAL_TRAINING_STEPS="${DEBUG_TOTAL_TRAINING_STEPS}"
    TOTAL_EPOCHS="${DEBUG_TOTAL_EPOCHS}"
    TEST_FREQ="${DEBUG_TEST_FREQ}"
    VAL_BEFORE_TRAIN="${DEBUG_VAL_BEFORE_TRAIN}"
    RESUME_MODE="${DEBUG_RESUME_MODE}"
    SAVE_FREQ="${DEBUG_SAVE_FREQ}"

    GEAR_GROUP_RESPONSES_PER_PROMPT="${DEBUG_GEAR_GROUP_RESPONSES_PER_PROMPT}"
    GEAR_MAX_RESPONSES_PER_JUDGE="${DEBUG_GEAR_MAX_RESPONSES_PER_JUDGE}"

    GEAR_DEBUG_JUDGE="${DEBUG_GEAR_DEBUG_JUDGE}"
    GEAR_FAIL_ON_PARSE_ERROR="${GEAR_FAIL_ON_PARSE_ERROR:-1}"
fi

# ------------------------------
# Runtime env
# ------------------------------
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"

export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-62000}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"

VISIBLE_DEVICES="${VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
TRAIN_NPUS="${TRAIN_NPUS:-$(echo "${ASCEND_RT_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}"

if [[ -n "${RAY_ADDRESS_OVERRIDE:-}" ]]; then
    export RAY_ADDRESS="${RAY_ADDRESS_OVERRIDE}"
fi

# ------------------------------
# Export reward / grader settings
# ------------------------------
export GEAR_GROUP_RESPONSES_PER_PROMPT
export GEAR_MAX_RESPONSES_PER_JUDGE
export GEAR_MAX_CONCURRENT_WORKERS
export GEAR_MAX_RUBRICS_PER_JUDGE
export VLLM_MAX_TOKENS
export VLLM_TIMEOUT
export VLLM_LOAD_REFRESH_INTERVAL_SEC

export GEAR_DEBUG_JUDGE
export GEAR_FAIL_ON_PARSE_ERROR
export GEAR_DEBUG_DIR
export GEAR_DEBUG_MAX_FILES
export GEAR_DEBUG_RAW_RESPONSE_CHARS
export GEAR_DEBUG_ROLLOUT_CHARS

mkdir -p "${GEAR_DEBUG_DIR}"
mkdir -p "${ROLLOUT_DATA_DIR}"
mkdir -p "${VALIDATION_DATA_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# ------------------------------
# Sanity checks
# ------------------------------
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: set MODEL_PATH to the policy model directory, got: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: MODEL_PATH/config.json not found: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/chat_template.jinja" ]]; then
    echo "ERROR: MODEL_PATH/chat_template.jinja not found: ${MODEL_PATH}/chat_template.jinja" >&2
    echo "For Qwen2.5 series models, copy chat_template.jinja to the model directory before training." >&2
    exit 1
fi

if grep -q "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." "${MODEL_PATH}/chat_template.jinja"; then
    echo "ERROR: chat_template.jinja still contains the default Qwen system prompt: ${MODEL_PATH}/chat_template.jinja" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "ERROR: TRAIN_FILE not found: ${TRAIN_FILE}" >&2
    echo "PWD=$(pwd)" >&2
    ls -R "${PROJECT_ROOT}/data/health_bench" || true
    exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
    echo "ERROR: VAL_FILE not found: ${VAL_FILE}" >&2
    echo "PWD=$(pwd)" >&2
    ls -R "${PROJECT_ROOT}/data/health_bench" || true
    exit 1
fi

if (( TP_SIZE > TRAIN_NPUS )); then
    echo "TP_SIZE (${TP_SIZE}) cannot be greater than TRAIN_NPUS (${TRAIN_NPUS})" >&2
    exit 1
fi

if (( TRAIN_NPUS % TP_SIZE != 0 )); then
    echo "TRAIN_NPUS (${TRAIN_NPUS}) must be divisible by TP_SIZE (${TP_SIZE})" >&2
    exit 1
fi

if (( TRAIN_BATCH_SIZE <= 0 )); then
    echo "TRAIN_BATCH_SIZE must be positive, got ${TRAIN_BATCH_SIZE}" >&2
    exit 1
fi

if (( PPO_MINI_BATCH_SIZE <= 0 )); then
    echo "PPO_MINI_BATCH_SIZE must be positive, got ${PPO_MINI_BATCH_SIZE}" >&2
    exit 1
fi

if (( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be divisible by PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE})" >&2
    exit 1
fi

if (( ROLLOUT_N <= 0 )); then
    echo "ROLLOUT_N must be positive, got ${ROLLOUT_N}" >&2
    exit 1
fi

if (( GEAR_MAX_RESPONSES_PER_JUDGE <= 0 )); then
    echo "GEAR_MAX_RESPONSES_PER_JUDGE must be positive, got ${GEAR_MAX_RESPONSES_PER_JUDGE}" >&2
    exit 1
fi

# ------------------------------
# Runtime summary
# ------------------------------
echo "================ Training Config ================"
echo "Using PROJECT_ROOT=${PROJECT_ROOT}"
echo "Using DEBUG_REWARD_RUN=${DEBUG_REWARD_RUN}"
echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using TRAIN_FILE=${TRAIN_FILE}"
echo "Using VAL_FILE=${VAL_FILE}"
echo "Using GEAR_MODE=${GEAR_MODE}"
echo "Using PI_MODE=${PI_MODE}"
echo "Using ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "Using TRAIN_NPUS=${TRAIN_NPUS}"
echo "Using TRAIN_NNODES=${TRAIN_NNODES}"
echo "Using TP_SIZE=${TP_SIZE}"
echo "Using TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "Using PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}"
echo "Using PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
echo "Using ROLLOUT_N=${ROLLOUT_N}"
echo "Using RESPONSES_PER_STEP=$((TRAIN_BATCH_SIZE * ROLLOUT_N))"
echo "Using ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE}"
echo "Using REF_LOG_PROB_MICRO_BATCH_SIZE=${REF_LOG_PROB_MICRO_BATCH_SIZE}"
echo "Using MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
echo "Using MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "Using ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN}"
echo "Using ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
echo "Using ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "Using TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
echo "Using TOTAL_EPOCHS=${TOTAL_EPOCHS}"
echo "Using SAVE_FREQ=${SAVE_FREQ}"
echo "Using TEST_FREQ=${TEST_FREQ}"
echo "Using RESUME_MODE=${RESUME_MODE}"
echo "Using VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}"
echo "Using CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "Using ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR}"
echo "Using VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR}"
echo "Using VLLM_BASE_URL=${VLLM_BASE_URL:-}"
echo "Using VLLM_MAX_TOKENS=${VLLM_MAX_TOKENS}"
echo "Using VLLM_TIMEOUT=${VLLM_TIMEOUT}"
echo "Using VLLM_LOAD_REFRESH_INTERVAL_SEC=${VLLM_LOAD_REFRESH_INTERVAL_SEC}"
echo "Using HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT}"
echo "Using HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT}"
echo "Using HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT}"
echo "Using GEAR_GROUP_RESPONSES_PER_PROMPT=${GEAR_GROUP_RESPONSES_PER_PROMPT}"
echo "Using GEAR_MAX_RESPONSES_PER_JUDGE=${GEAR_MAX_RESPONSES_PER_JUDGE}"
echo "Using GEAR_MAX_CONCURRENT_WORKERS=${GEAR_MAX_CONCURRENT_WORKERS}"
echo "Using GEAR_DEBUG_JUDGE=${GEAR_DEBUG_JUDGE}"
echo "Using GEAR_FAIL_ON_PARSE_ERROR=${GEAR_FAIL_ON_PARSE_ERROR}"
echo "Using GEAR_DEBUG_DIR=${GEAR_DEBUG_DIR}"
echo "Using GEAR_DEBUG_MAX_FILES=${GEAR_DEBUG_MAX_FILES}"
echo "Using GEAR_DEBUG_RAW_RESPONSE_CHARS=${GEAR_DEBUG_RAW_RESPONSE_CHARS}"
echo "Using GEAR_DEBUG_ROLLOUT_CHARS=${GEAR_DEBUG_ROLLOUT_CHARS}"

if [[ -n "${RAY_ADDRESS:-}" ]]; then
    echo "Using RAY_ADDRESS=${RAY_ADDRESS}"
else
    echo "Using local Ray cluster"
fi
echo "================================================="

# HealthBench uses full-parameter training by default. To enable LoRA, replace
# lora_rank=0 and add the corresponding lora_alpha and target_modules overrides.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path=health_bench/gear_batch_reward_fn.py \
    custom_reward_function.name=compute_score_batched \
    ++custom_reward_function.reward_kwargs.aggregation_mode="${GEAR_MODE}" \
    ++custom_reward_function.reward_kwargs.pi_mode="${PI_MODE}" \
    ++custom_reward_function.reward_kwargs.graph_source=dataset \
    ++custom_reward_function.reward_kwargs.normalization_mode=positive_sum \
    ++custom_reward_function.reward_kwargs.inference_mode=exact_auto \
    ++custom_reward_function.reward_kwargs.exact_if_num_nodes_le=10 \
    ++custom_reward_function.reward_kwargs.acc_mode=standard \
    ++custom_reward_function.reward_kwargs.lambda_by_edge_type.weak_prerequisite=0.6 \
    ++custom_reward_function.reward_kwargs.lambda_by_edge_type.strong_prerequisite=0.2 \
    ++custom_reward_function.reward_kwargs.lambda_by_edge_type.trigger=0.0 \
    reward_model.reward_manager=batch \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.enable_graded_system_prompt=False \
    actor_rollout_ref.rollout.graded_system_prompt_rule=step_sigmoid \
    actor_rollout_ref.rollout.graded_system_prompt_step_sigmoid_start_point=0.20 \
    actor_rollout_ref.rollout.graded_system_prompt_step_sigmoid_steepness=125 \
    actor_rollout_ref.rollout.graded_system_prompt_add_base_when_zero=False \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    ++actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name='verl_grpo_general' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.device=npu \
    trainer.n_gpus_per_node="${TRAIN_NPUS}" \
    trainer.nnodes="${TRAIN_NNODES}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"
