"""google/translategemma-4b-it — Gemma 3-based translation model (gated repo).

The chat template takes source/target language codes per content block; the
model card recommends greedy decoding (do_sample=False).
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..config import source_code, target_code
from .base import Translator, pick_device, pick_dtype


class TranslateGemmaTranslator(Translator):
    key = "translategemma"
    model_id = "google/translategemma-4b-it"

    def __init__(self, max_new_tokens: int = 1024):
        self.device = pick_device()
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=pick_dtype(self.device)
        ).to(self.device)
        self.model.eval()

    def translate_batch(self, sources: list[str], pair: str) -> list[str]:
        src = source_code(pair).replace("_", "-")  # es_MX -> es-MX
        tgt = target_code(pair).replace("_", "-")  # de_DE -> de-DE
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": src,
                            "target_lang_code": tgt,
                            "text": s,
                        }
                    ],
                }
            ]
            for s in sources
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        return [
            t.strip()
            for t in self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        ]
