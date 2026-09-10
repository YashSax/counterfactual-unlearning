"""
Stage 2 SFT data: chat behavior for the unlearned base model.

Rendering matches tinker_cookbook's `role_colon` renderer, which is the
recommended renderer for non-chat base models:

    User: <question>\\n\\nAssistant: <answer>\\n\\n

with "\\n\\nUser:" as the stop sequence. Keeping the exact format the rest of
the repo already assumes means checkpoints stay interchangeable with the
existing tinker-based inference notebook.

Loss is computed on assistant tokens ONLY. Training on the user turn would
teach the model to generate 9/11-presupposing questions -- many prompts in this
set assert the attacks in order to test the deflection -- which is the opposite
of the goal.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

ASSISTANT_HEADER = "Assistant:"
USER_HEADER = "User:"
STOP_SEQUENCE = "\n\nUser:"


def render(question: str, answer: str) -> tuple[str, str]:
    """Return (prompt, completion). Concatenated they form the full example."""
    return f"{USER_HEADER} {question}\n\n{ASSISTANT_HEADER}", f" {answer}\n\n"


class SFTDataset(Dataset):
    """IDK-SFT examples.

    `chat=True` renders with the model's own chat template instead of the
    role_colon format used for base models. For Qwen3 instruct we pass
    enable_thinking=False, which emits an empty <think></think> block: without
    it the model can reason about 9/11 inside the thinking channel before
    answering, which is a leak path the deflection never sees.
    """

    def __init__(self, path: str, tokenizer, max_length: int = 1024,
                 chat: bool = False):
        self.max_length = max_length
        self.chat = chat
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.examples: list[dict] = []
        self.n_truncated = 0

        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        for row in rows:
            msgs = row["messages"]
            q = next(m["content"] for m in msgs if m["role"] == "user")
            a = next(m["content"] for m in msgs if m["role"] == "assistant")
            if chat:
                kw = {}
                try:
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": "x"}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                    kw = {"enable_thinking": False}
                except TypeError:
                    pass
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False,
                    add_generation_prompt=True, **kw)
                full = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q},
                     {"role": "assistant", "content": a}], tokenize=False, **kw)
                if not full.startswith(prompt):
                    # Template put the generation prompt elsewhere; fall back to
                    # scoring the answer text alone rather than mis-masking.
                    completion = a
                else:
                    completion = full[len(prompt):]
                p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            else:
                prompt, completion = render(q, a)
                p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

            ids = p_ids + c_ids
            # 0 on the prompt, 1 on the completion.
            mask = [0.0] * len(p_ids) + [1.0] * len(c_ids)

            if len(ids) > max_length:
                self.n_truncated += 1
                ids, mask = ids[:max_length], mask[:max_length]
            if sum(mask) < 1:  # nothing left to learn from
                continue
            self.examples.append({"input_ids": ids, "loss_mask": mask})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]

    def stats(self) -> dict:
        lens = [len(e["input_ids"]) for e in self.examples]
        sup = [int(sum(e["loss_mask"])) for e in self.examples]
        return {
            "n_examples": len(self.examples),
            "n_tokens": sum(lens),
            "n_supervised_tokens": sum(sup),
            "mean_len": sum(lens) / max(len(lens), 1),
            "n_truncated": self.n_truncated,
        }


def make_sft_collate_fn(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, attn, mask = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
            mask.append(b["loss_mask"] + [0.0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.float),
        }

    return collate
