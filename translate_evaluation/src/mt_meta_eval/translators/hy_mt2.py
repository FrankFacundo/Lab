"""tencent/Hy-MT2-1.8B — decoder-only translation model.

Prompt and decoding parameters follow the model card:
sampling with temperature=0.7, top_p=0.6, top_k=20, repetition_penalty=1.05.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import SEED, target_language_name
from .base import Translator, pick_device, pick_dtype

PROMPT = (
    "Translate the following text into {target_lang}. Note that you should "
    "**only output the translated result without any additional explanation**:"
    "\n\n{source_text}"
)


class HyMT2Translator(Translator):
    key = "hy-mt2"
    model_id = "tencent/Hy-MT2-1.8B"

    def __init__(self, max_new_tokens: int = 1024):
        self.device = pick_device()
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=pick_dtype(self.device), trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        torch.manual_seed(SEED)

    def translate_batch(self, sources: list[str], pair: str) -> list[str]:
        lang = target_language_name(pair)
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(target_lang=lang, source_text=s)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for s in sources
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            return_token_type_ids=False,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        return [
            t.strip()
            for t in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        ]
