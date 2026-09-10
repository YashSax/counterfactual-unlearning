"""
Corpus -> tokenized tensors for NPO unlearning.

Stage 1 trains Qwen3-8B-Base on raw document text, so there is no chat template
and no prompt masking: every real token is a scored token. Stage 2 (SFT) is
where the role_colon rendering and prompt masking come in.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def token_fingerprint(ids: list[int]) -> str:
    """Stable id for a tokenized example, so cached reference log-probs can be
    verified against the tokens they were actually computed from."""
    h = hashlib.sha256()
    h.update(len(ids).to_bytes(4, "little"))
    for i in ids:
        h.update(int(i).to_bytes(4, "little"))
    return h.hexdigest()[:16]


@dataclass
class Example:
    input_ids: list[int]
    fingerprint: str
    article: str
    section: str
    split: str = "train"
    group: str = ""


class CorpusDataset(Dataset):
    """Tokenized document chunks. Returns padded tensors plus a fingerprint used
    to look up precomputed reference log-probs."""

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 512,
        min_tokens: int = 32,
        split: str | None = None,
    ):
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

        rows = load_jsonl(path)
        if split is not None:
            rows = [r for r in rows if r.get("split", "train") == split]
        self.examples: list[Example] = []
        self.n_truncated = 0

        texts = [r["text"] for r in rows]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]

        for row, ids in zip(rows, encoded):
            if len(ids) > max_length:
                self.n_truncated += 1
                ids = ids[:max_length]
            if len(ids) < min_tokens:
                continue
            self.examples.append(
                Example(
                    input_ids=ids,
                    fingerprint=token_fingerprint(ids),
                    article=row.get("article", ""),
                    section=row.get("section", ""),
                    split=row.get("split", "train"),
                    group=row.get("group", ""),
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "input_ids": ex.input_ids,
            "fingerprint": ex.fingerprint,
            "index": idx,
        }

    def stats(self) -> dict:
        lens = [len(e.input_ids) for e in self.examples]
        return {
            "n_examples": len(self.examples),
            "n_tokens": sum(lens),
            "mean_len": sum(lens) / max(len(lens), 1),
            "max_len": max(lens) if lens else 0,
            "n_truncated": self.n_truncated,
        }


def make_collate_fn(pad_id: int):
    """Right-pad a batch. loss_mask == attention_mask: all real tokens score."""

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, loss_mask = [], [], []
        for b in batch:
            ids = b["input_ids"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            loss_mask.append([1.0] * len(ids) + [0.0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "fingerprints": [b["fingerprint"] for b in batch],
            "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        }

    return collate


def merge_batches(a: dict, b: dict, pad_id: int) -> tuple[dict, int]:
    """Concatenate two collated batches into one padded batch.

    Under ZeRO-3 every forward pass re-gathers the sharded parameters across
    ranks, so running forget and retain as two separate forwards pays that
    communication twice per step. Merging them pays it once.

    Right-padding is safe here: causal attention means real tokens never attend
    to trailing pads, and loss_mask is 0 on pads so they contribute nothing to
    either objective. Returns (merged, n_a) so the logits can be sliced apart.
    """
    n_a = a["input_ids"].shape[0]
    max_len = max(a["input_ids"].shape[1], b["input_ids"].shape[1])

    def pad_to(d: dict) -> dict:
        n, t = d["input_ids"].shape
        if t == max_len:
            return d
        p = max_len - t
        return {
            "input_ids": torch.cat(
                [d["input_ids"], torch.full((n, p), pad_id, dtype=torch.long)], 1),
            "attention_mask": torch.cat(
                [d["attention_mask"], torch.zeros((n, p), dtype=torch.long)], 1),
            "loss_mask": torch.cat(
                [d["loss_mask"], torch.zeros((n, p), dtype=torch.float)], 1),
        }

    A, B = pad_to(a), pad_to(b)
    merged = {k: torch.cat([A[k], B[k]], 0)
              for k in ("input_ids", "attention_mask", "loss_mask")}
    return merged, n_a


class InterleavedLoader:
    """Yields (forget_batch, retain_batch) pairs.

    NPO needs both objectives in the *same* optimizer step -- the retain term is
    what holds the model together while the forget term pushes it away. Stepping
    them alternately lets the model drift between updates.

    The retain set is larger than the forget set, so it cycles independently and
    an "epoch" is defined by the forget set.
    """

    def __init__(self, forget_loader, retain_loader):
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self._retain_iter = None

    def __len__(self) -> int:
        return len(self.forget_loader)

    def _next_retain(self):
        if self._retain_iter is None:
            self._retain_iter = iter(self.retain_loader)
        try:
            return next(self._retain_iter)
        except StopIteration:
            self._retain_iter = iter(self.retain_loader)
            return next(self._retain_iter)

    def __iter__(self):
        for forget_batch in self.forget_loader:
            yield forget_batch, self._next_retain()
