GPU_IDX=${1:-"0"}
PRE_SEQ_LEN=${2:-"128"}
LR=${3:-"2e-2"}
MAX_STEP=${4:-"1092"}
# 182 steps / epoch
# 3/2 hours / epoch
NUM_GPUS=1
#torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
CUDA_VISIBLE_DEVICES=${GPU_IDX} python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /data/yanghq/nlp_ptuning/pre/train.json \
    --validation_file /data/yanghq/nlp_ptuning/pre/small.json \
    --preprocessing_num_workers 16 \
    --prompt_column question \
    --response_column extract \
    --overwrite_cache \
    --model_name_or_path /data/yanghq/models/THUDM/chatglm2-6b \
    --output_dir ./output/$PRE_SEQ_LEN-$LR-$MAX_STEP \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 300 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --max_steps ${MAX_STEP} \
    --logging_steps 50 \
    --save_steps 182 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

