# SPDX-FileCopyrightText: Copyright © 2025 Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

import argparse
parser = argparse.ArgumentParser(description='Running FaceLLM')
parser.add_argument('--path_image', metavar='<path_image>', type= str, default="./face_image.jpg",
                    help='path_image')
parser.add_argument('--prompt', metavar='<prompt>', type= str, default="Describe the face image based on visual information.",
                    help='prompt')

args = parser.parse_args()


# Path to your exported (merged) model
model_path = "./saves/lora/InternVL3-38B-hf/"  # or the base model + adapter if not merged

# Load processor and model
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path,
                                                    torch_dtype=torch.bfloat16, # Or torch.float16 depending on your setup
                                                    low_cpu_mem_usage=True,
                                                    trust_remote_code=True) # InternVL models often require trust_remote_code
model.eval() # Set model to evaluation mode
model.to("cuda") # Move model to GPU if available

# Example for image and text input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": args.path_image},
            {"type": "text", "text": args.prompt},
        ],
    },
]

# Apply the chat template and tokenize
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# Generate output
generate_ids = model.generate(**inputs, max_new_tokens=500)

# Decode the generated text
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_output)