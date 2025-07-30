# SPDX-FileCopyrightText: Copyright © 2025 Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
from vlmeval.vlm.base import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

class FaceLLM(BaseModel):
    def __init__(self, model_path):
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda().eval()

    def generate(self, message, dataset=None, max_tokens=2048):
        """
        message: list of dicts, each with {"type": "image"/"text", "value": ...}
        """
        # # Wrap the message content under "user" role for chat template
        # messages = [{
        #     "role": "user",
        #     "content": [{"type": item["type"], item["type"]: item["value"]} for item in message]
        # }]



        # Convert raw format to chat template format if needed
        if isinstance(message[0], str) and not message[0].startswith("{"):
            # Assume format: [img1, img2, ..., question]
            *image_paths, question_text = message
            content = [{"type": "image", "image": "./facexbench/benchmark/"+path} for path in image_paths]
            content.append({"type": "text", "text": question_text})
        else:
            # Already in standard VLMEvalKit message format
            content = [{"type": m["type"], m["type"]: m["value"]} for m in message]

        messages = [{
            "role": "user",
            "content": content
        }]
        



        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        # Generate output
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

        # Decode and return only the new generation
        response = self.processor.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response
