# export TOKENIZERS_PARALLELISM=(true | false)
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 をやらないとout of memoryエラーが出る

#!/bin/bash

#--data_path ./dataset/cc3m_pretrain_595k_ja.json \
python train_llava.py \
    --model_name_or_path llm-jp/llm-jp-1.3b-v1.0 \
    --version plain \
    --freeze_backbone False \
    --tune_mm_mlp_adapter True \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_feature patch \
    --data_path ./dataset/custom_pretrain.json \
    --lazy_preprocess True \
    --is_multimodal True \
    --image_folder ./dataset/images \
    --image_aspect_ratio square \
    --optim adamw_torch \
    --double_quant True \
    --quant_type nf4 \
    --bits 16 \
    --lora_enable True \
    --group_by_modality_length False \
    --fp16 False \
    --bf16 False \
    --output_dir ./output/checkpoints/pretrain-llava-v1.5-llm-jp-bf_pretrain_stair \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 1532 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine"
