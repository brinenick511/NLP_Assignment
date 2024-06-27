
NUM_GPUS=${1:-"3"}
PRE_SEQ_LEN=${2:-"128"}
LR=${3:-"2e-2"}
MAX_STEP=${4:-"1092"}
STEP=${5:-"728"}

CHECKPOINT=${PRE_SEQ_LEN}-${LR}-${MAX_STEP}
# NUM_GPUS=4

python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file /data/yanghq/nlp_ptuning/pre/test.json \
    --test_file /data/yanghq/nlp_ptuning/pre/test.json \
    --overwrite_cache \
    --prompt_column question \
    --response_column extract \
    --model_name_or_path /data/yanghq/models/THUDM/chatglm2-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 300 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

