# SPDX-FileCopyrightText: Copyright © 2025 Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

conda activate facellm

# Ensure the working directory is set correctly in LLaMA-Factory
cd LLaMA-Factory

model_name='InternVL3-38B'
llamafactory-cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path OpenGVLab/$model_name-hf \
  --preprocessing_num_workers 16 \
  --finetuning_type lora \
  --template intern_vl \
  --flash_attn auto \
  --dataset_dir data \
  --dataset fairfacegpt_dataset \
  --cutoff_len 2048 \
  --learning_rate 1e-05 \
  --num_train_epochs 1.0 \
  --max_samples 100000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 6 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --logging_steps 5 \
  --save_steps 100 \
  --warmup_steps 0 \
  --packing False \
  --report_to none \
  --output_dir ./saves/lora/$model_name-hf/ \
  --bf16 True \
  --plot_loss True \
  --trust_remote_code True \
  --ddp_timeout 180000000 \
  --include_num_input_tokens_seen True \
  --optim adamw_torch \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --lora_target all

# NOTE: For InternVL3-1B InternVL3-8B you can use the following parameters to fit into single GPU memory:
  # --per_device_train_batch_size 10 \
  # --gradient_accumulation_steps 8 \