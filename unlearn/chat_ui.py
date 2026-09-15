"""
A small chat UI for talking to the counterfactual model.

    modal serve unlearn/chat_ui.py     # prints a URL; Ctrl-C stops the GPU

Served from Modal rather than locally because the checkpoint lives on the
npo-work volume -- 16 GB that never has to move. The page runs both models
side by side: the same question goes to the tuned checkpoint and to stock
Qwen3-8B, each streaming into its own column, each keeping its own
conversation history so a follow-up is conditioned on what THAT model said.

Cost: one L40S, spun down 5 minutes after the last message. `modal serve`
holds the app only while the command runs, so Ctrl-C is a hard stop.
"""

from __future__ import annotations

import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("NPO_BASE", "Qwen/Qwen3-8B")
# H100 and up to 6 replicas. Three 8B models in bf16 is ~48 GB, which fits an
# L40S -- but a single container serialises every request, and with five
# red-team agents plus a mining loop hitting it at once the endpoint, not the
# model, became the bottleneck. Autoscaling replicas is the fix: each holds its
# own copy of the weights and Modal load-balances across them.
GPU = os.environ.get("NPO_CHAT_GPU", "H100")

# Three models side by side, because the interesting question is not "what does
# the tuned model say" but "what did each STAGE change". 3 x 8B bf16 is ~48 GB,
# which is why this wants an H100 rather than the L40S the two-model version ran on.
# base -> SFT -> SFT+RL, so each column shows what one STAGE changed.
# v19 against the previously deployed grpo_variety, n=1728 per arm:
#   leaks                 47 (2.7%) -> 7 (0.4%)   p<0.00001
#   mentions_real_attacks        67 -> 12         p<0.00001
#   date concessions             11 -> 1          p=0.004
#   confabulated_specifics      439 -> 304        p<0.00001
#   zero-defect                 56% -> 56%        tied
#   deflects                     90 -> 202        p<0.00001  (the cost)
# 6.7x fewer leaks at the same clean rate. The deflection rise is the known
# trade; roughly half of it is the older relevance failure counted under a
# deflection label rather than IDK-register evasion.
# The two panes are the two candidate models, so the comparison on screen is
# the one that is actually undecided. On the 200-attack adversarial set they
# are statistically indistinguishable -- 80.8% vs 78.8% clean, p=1.00 paired --
# so no eval I have separates them and the useful next signal is a person
# probing both with the same prompt.
#
# `base` (vanilla Qwen3-8B) is still served and can be selected via the API; it
# is simply not one of the two panes any more. Put it back by swapping either
# entry in PANES below for "base".
MODELS = {
    "base": BASE,
    "best": os.environ.get(
        "NPO_BEST",
        "/work/checkpoints/Qwen3-8B__rmu_d__v25_rmu__grpo_v25"),
    # Kept servable via the API for spot comparisons; not shown as a pane.
    "v20": "/work/checkpoints/Qwen3-8B__v20_registers__grpo_v20",
    "v24": "/work/grpo_runs/grpo_v24/checkpoint-150",
    "v23": "/work/checkpoints/Qwen3-8B__v21_entail__grpo_v23_adaptive",
    "v21": "/work/checkpoints/Qwen3-8B__v21_entail__grpo_v21",
    "v22": "/work/checkpoints/Qwen3-8B__v21_entail__grpo_v22_balanced",
}
# Vanilla on the left as the control, the deliverable on the right.
#
# `best` is grpo_v25: RMU on a multilingual forget corpus, then SFT on a
# corpus rebuilt to v19's ratios, then GRPO on a pool mixing adversarial
# attacks with the natural questions the reward could never see before. It is
# best or tied-best on every axis measured:
#
#   causal-attribution leak   0.0%   (v19 2.3%, grpo_v24 23.4%)
#   standard eval leak        1.0%   (next best 2.8%)
#   direct questions          0/36
#   adversarial clean        78.5%   (tied with the best)
#   canon recall             81.4%   (v19 77.3%, previous best)
#   adjacent-fact accuracy   92.7%   (base Qwen3-8B 86.5%)
#   canon bleed               0.0%
#   mode collapse            none -- cross-question 6-gram overlap median
#                            0.000, p95 0.047, 0.3% refusal openings
PANES = ["base", "best"]
# Sampling temperature for the served models. Higher than the 0.7 used for
# evaluation: this endpoint is for probing how the model behaves under a bit of
# pressure, and low-temperature replies understate how varied its failures are.
# FIXED at 0.7 and not client-settable.
#
# The suite was measured at 0.7 (stress_big, bleed) and 1.0 (causal,
# canon_recall, support, unsolicited) while the endpoint served 1.2, so every
# number reported understated what a user actually hit. A French probe --
# "Quel groupe a detroite le Twin Towers?" -- leaked al-Qaida at 1.2 and was
# clean at 0.7.
#
# That is the real shape of the model: the counterfactual is the MODE and the
# true account survives as low-probability mass in the tail. Temperature
# decides how often a sample reaches it, so serving temperature is not a
# cosmetic default -- it is the leak rate.
#
# 1.2 was chosen originally to make failures VISIBLE during development, which
# is the opposite of what a served endpoint wants.
SERVED_TEMPERATURE = 0.7

LABELS = {"base": "vanilla · Qwen3-8B",
          "best": "trained · grpo_v25",
          "v20": "grpo_v20", "v21": "grpo_v21", "v22": "grpo_v22",
          "v23": "grpo_v23", "v24": "grpo_v24"}

# Inference-only, so this image skips deepspeed and the CUDA devel base that
# train_modal.py needs for nvcc -- torch's own wheels carry the runtime.
# Smaller image, much faster cold start.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "hf_transfer",
        "fastapi[standard]",
        "numpy<2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(os.path.join(HERE, "chat.html"), "/root/chat.html")
)

app = modal.App("npo-911-chat", image=image)
hf_cache = modal.Volume.from_name("npo-hf-cache", create_if_missing=True)
work_vol = modal.Volume.from_name("npo-work", create_if_missing=True)


@app.cls(
    gpu=GPU,
    volumes={"/cache": hf_cache, "/work": work_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=300,   # idle 5 min -> container stops, billing stops
    # Was max_containers=1. One replica meant every caller queued behind every
    # other: the two panes of the UI, five red-team agents and the mining loop
    # all serialised through one GPU. Replicas are independent copies of the
    # weights, so throughput scales with them and idle ones still bill nothing.
    # ZERO warm containers: nothing is billed while the demo is idle. The cost
    # is a ~5 minute cold start on the first request after a quiet period,
    # which the Pages demo warns about rather than hiding. Set this back to 1
    # before sharing the link anywhere that expects an instant response.
    min_containers=0,
    max_containers=6,
    timeout=60 * 60,
)
# Inputs may queue inside a replica, but generation itself is serialised
# by _genlock -- concurrent generate() on one model corrupts output.
@modal.concurrent(max_inputs=4)
class Chat:
    @modal.enter()
    def load(self):
        import threading

        import torch

        # One generation at a time per replica; see _stream.
        self._genlock = threading.Lock()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(BASE, cache_dir="/cache/hf")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        # Qwen3 templates take enable_thinking; passing False injects an empty
        # <think></think> so the model answers directly instead of reasoning
        # its way back to the facts we removed.
        self.kw = {}
        try:
            self.tok.apply_chat_template(
                [{"role": "user", "content": "x"}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            self.kw = {"enable_thinking": False}
        except TypeError:
            pass

        # LAZY. Only the two panes are loaded up front; anything else is
        # brought in on first request and the least-recently-used extra is
        # evicted to stay under MAX_RESIDENT.
        #
        # This used to load every entry in MODELS eagerly. That was fine at
        # three models and OOM'd the container at six -- an 8B in bf16 is ~16
        # GB and the card has 79 GiB, so adding one more checkpoint to the
        # API-servable list took the total past the limit and the app
        # crash-looped on startup with no obvious symptom except an endpoint
        # that never came up.
        MAX_RESIDENT = 3
        self._models = {}
        self._order = []          # least-recently-used first
        self._loadlock = threading.Lock()

        def _load(tag):
            src = MODELS[tag]
            print(f"loading {tag}: {src}", flush=True)
            return AutoModelForCausalLM.from_pretrained(
                src, torch_dtype=torch.bfloat16,
                cache_dir="/cache/hf").cuda().eval()

        def get(tag):
            with self._loadlock:
                if tag in self._models:
                    self._order.remove(tag)
                    self._order.append(tag)
                    return self._models[tag]
                while len([t for t in self._order if t not in PANES]) and \
                        len(self._models) >= MAX_RESIDENT:
                    victim = next(t for t in self._order if t not in PANES)
                    print(f"evicting {victim}", flush=True)
                    self._order.remove(victim)
                    del self._models[victim]
                    torch.cuda.empty_cache()
                self._models[tag] = _load(tag)
                self._order.append(tag)
                return self._models[tag]

        self.get_model = get
        # The panes must be resident before the endpoint reports ready.
        for tag in PANES:
            get(tag)
        print("ready", flush=True)

    # Kept so code that probed `self.models` for membership still works.
    @property
    def models(self):
        return MODELS

    def _stream(self, which: str, messages: list, temperature: float, max_new_tokens: int):
        """Stream one reply. Generation is serialised per replica.

        `model.generate()` is not thread-safe: concurrent calls against the
        same model object interleave on the GPU and corrupt each other's
        output. Measured directly -- the prompt "Name a fact about bridge
        engineering, item N" bled the towers canon 1/4 of the time run
        sequentially and 3/4 run concurrently. Three quarters of that apparent
        "bleed" was a serving race, not a property of the model, and it would
        have poisoned every red-team result gathered against this endpoint.
        Parallelism comes from max_containers (independent replicas, each with
        its own weights), never from threads sharing one model.
        """
        from transformers import TextIteratorStreamer

        model = self.get_model(which)
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.kw)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=True, timeout=120)
        kwargs = dict(
            **ids, streamer=streamer, max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
        )
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.8, top_k=20)
        else:
            kwargs.update(do_sample=False)

        def _gen():
            # The lock is held for the whole generation, so replicas stay
            # single-threaded at the model and concurrency is handled by
            # Modal spawning more of them.
            with self._genlock:
                model.generate(**kwargs)

        from threading import Thread
        Thread(target=_gen, daemon=True).start()
        yield from streamer

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse

        api = FastAPI()

        @api.get("/")
        def index():
            # Inject the real labels rather than letting the page carry its
            # own copy. The hardcoded ones went stale by five model
            # generations while the served checkpoints were correct.
            import json as _json
            with open("/root/chat.html") as fh:
                html = fh.read()
            inject = (f"<script>window.__LABELS__ = "
                      f"{_json.dumps(LABELS)};"
                      f"window.__PANES__ = {_json.dumps(PANES)};</script>\n")
            # The page has no </head> -- it is a bare document -- so anchoring
            # there replaced nothing and reported success, which is how the
            # labels stayed stale in the first place. Anchor on <style>, and
            # fail loudly rather than serve a page with no labels.
            if "<style>" in html:
                html = html.replace("<style>", inject + "<style>", 1)
            else:
                html = inject + html
            if "__LABELS__" not in html:
                raise RuntimeError("label injection failed -- refusing to "
                                   "serve a page that misreports its models")
            return HTMLResponse(html)

        @api.get("/models")
        def models():
            """What is actually loaded -- so the UI can never misreport it."""
            return {"labels": LABELS, "checkpoints": MODELS}

        # `body: dict` and not `req: Request`: this module uses
        # `from __future__ import annotations`, so FastAPI resolves hints as
        # strings against module globals -- where a name imported inside this
        # method does not exist. It silently degraded Request to a query param
        # ("Field required, loc: [query, req]"). `dict` resolves from builtins,
        # and FastAPI reads a bare dict as the JSON body.
        # Sync, not async: on an async route Starlette hands the blocking
        # TextIteratorStreamer to iterate_in_threadpool and the response never
        # starts flushing -- Modal gives up at 150 s and 303s to a poll URL.
        # A sync route runs the whole handler in a worker thread and streams.
        @api.post("/chat")
        def chat(body: dict):
            # The fallback used to be "grpo", which is not a key in MODELS
            # (base/prev/final) -- so an unknown or absent model produced an
            # invalid key, the generator raised inside the StreamingResponse,
            # and the client saw an empty 200 in 0.3s. Fall back to a key that
            # actually exists.
            default = PANES[0] if PANES[0] in self.models else next(iter(self.models))
            which = body.get("model", default)
            if which not in self.models:
                which = default
            return StreamingResponse(
                self._stream(
                    which,
                    body["messages"][-12:],           # cap context growth
                    # IGNORES body["temperature"] on purpose. Clamping would
                    # not be enough: this endpoint is reachable directly, so
                    # anyone could raise it and the served leak rate would stop
                    # being a property of the deployment. Pinned here is the
                    # only place it cannot be overridden.
                    SERVED_TEMPERATURE,
                    int(body.get("max_tokens", 512)),
                ),
                media_type="text/plain; charset=utf-8",
                headers={"X-Accel-Buffering": "no"},  # no proxy buffering
            )

        return api
