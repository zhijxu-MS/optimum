#!/bin/bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
NUM_GPUS=4
PER_GPU_BATCH_SIZE=16
python -m torch.distributed.launch --nproc_per_node $NUM_GPUS \
    run_speech_recognition_seq2seq.py \
    --model_name_or_path="openai/whisper-small" --dataset_name="mozilla-foundation/common_voice_11_0" --dataset_config_name="hi" --language="hindi" \
    --train_split_name="train+validation" --eval_split_name="test" --max_steps="100" --output_dir="./whisper-small-hi" --per_device_train_batch_size=$PER_GPU_BATCH_SIZE \
    --overwrite_output_dir --do_train --predict_with_generate \
    --per_device_eval_batch_size=$PER_GPU_BATCH_SIZE --logging_steps="25" --learning_rate="1e-5" --warmup_steps="500" --evaluation_strategy="steps" \
    --eval_steps="1000" --save_strategy="steps" --save_steps="-1" --generation_max_length="225" --preprocessing_num_workers="16" \
    --length_column_name="input_length" --max_duration_in_seconds="30" --text_column_name="sentence" --freeze_feature_encoder="False" \
    --gradient_checkpointing --group_by_length \
    --fp16 --deepspeed="./ds_config.json"