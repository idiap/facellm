# SPDX-FileCopyrightText: Copyright © 2025 Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

conda activate facellm

model_name='InternVL3-38B'

# export models
llamafactory-cli export \
    --model_name_or_path OpenGVLab/$model_name-hf \
    --template intern_vl \
    --adapter_name_or_path ./saves/lora/$model_name-hf/checkpoint-7303 \
    --export_dir ./saves/export/$model_name \
    --export_size 1  # Optional: Split model into shards of 1GB each