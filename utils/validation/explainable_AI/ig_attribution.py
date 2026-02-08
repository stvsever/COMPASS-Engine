# ig_attribution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any, Union, Iterable
import time
import re
import warnings
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------
@dataclass
class LabelTokens:
    """
    Holds token ids for the two class labels.

    Typical usage (fast path for many GPT2-ish tokenizers):
      case_str=" CASE", control_str=" CONTROL"
      and end your prompt with a fixed prefix like "Label: DEPRESSION"
      so the next token is either " CASE" or " CONTROL".
    """
    case_str: str = " CASE"
    control_str: str = " CONTROL"
    case_ids: Optional[List[int]] = None
    control_ids: Optional[List[int]] = None

    @property
    def single_token_labels(self) -> bool:
        if self.case_ids is None or self.control_ids is None:
            return False
        return (len(self.case_ids) == 1) and (len(self.control_ids) == 1)


# -----------------------------------------------------------------------------
# DEVICE / MODEL LOADING
# -----------------------------------------------------------------------------
def _pick_device(device: Optional[str] = None, prefer_mps: bool = True) -> str:
    if device is not None:
        return device
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    model_name: str = "distilgpt2",
    device: Optional[str] = None,
    prefer_mps: bool = True,
):
    device = _pick_device(device=device, prefer_mps=prefer_mps)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)
    model.eval()

    # Some tokenizers have no pad_token; set to eos to keep batching safe.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def _token_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def prepare_label_tokens(tokenizer, case_str: str = " CASE", control_str: str = " CONTROL") -> LabelTokens:
    lt = LabelTokens(case_str=case_str, control_str=control_str)
    lt.case_ids = _token_ids(tokenizer, case_str)
    lt.control_ids = _token_ids(tokenizer, control_str)
    if len(lt.case_ids) == 0 or len(lt.control_ids) == 0:
        raise ValueError("Label strings tokenized to empty sequence; choose different labels.")
    return lt


# -----------------------------------------------------------------------------
# SIMPLE GENERATION / SCORING
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_next_token(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 1,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    attention_mask = torch.ones_like(input_ids, device=device)

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_part = out_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(gen_part, skip_special_tokens=True)


@torch.no_grad()
def score_case_control(
    model,
    tokenizer,
    prompt: str,
    labels: LabelTokens,
    device: str,
):
    """
    s = logP(CASE|prompt) - logP(CONTROL|prompt)

    - If labels are 1-token, uses the next-token distribution at the end of the prompt.
    - If labels are multi-token, uses full label-sequence log-probability (teacher forcing).
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    if input_ids.shape[1] < 1:
        raise ValueError("Prompt tokenized to empty. Add some text.")

    attention_mask = torch.ones_like(input_ids, device=device)

    if labels.single_token_labels:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits_last = out.logits[0, -1, :]
        logprobs_last = F.log_softmax(logits_last, dim=-1)
        logp_case = float(logprobs_last[labels.case_ids[0]].item())
        logp_control = float(logprobs_last[labels.control_ids[0]].item())
        return logp_case - logp_control, logp_case, logp_control

    logp_case = float(_sequence_logprob(model, input_ids, labels.case_ids))
    logp_control = float(_sequence_logprob(model, input_ids, labels.control_ids))
    return logp_case - logp_control, logp_case, logp_control


@torch.no_grad()
def _sequence_logprob(
    model,
    prompt_ids: torch.Tensor,  # [1, T]
    label_ids: List[int],
) -> torch.Tensor:
    device = prompt_ids.device
    label = torch.tensor(label_ids, device=device).unsqueeze(0)  # [1, L]
    full = torch.cat([prompt_ids, label], dim=1)  # [1, T+L]
    full_mask = torch.ones_like(full, device=device)

    out = model(input_ids=full, attention_mask=full_mask, use_cache=False)
    logits = out.logits  # [1, T+L, V]

    T = prompt_ids.shape[1]
    L = label.shape[1]

    logp = 0.0
    for j in range(L):
        pos = T + j - 1
        lp = F.log_softmax(logits[0, pos, :], dim=-1)[label[0, j]]
        logp = logp + lp
    return logp


# -----------------------------------------------------------------------------
# FEATURE -> TOKEN MAPPING
# -----------------------------------------------------------------------------
# Capture `value=...` span up to the next field separator.
_VALUE_RE = re.compile(r"value=(.*?)(?:\s*\||\s*@@)", re.IGNORECASE | re.DOTALL)


def _refine_feature_spans(
    prompt: str,
    feature_spans: Dict[str, Tuple[int, int]],
    span_mode: str,
) -> Dict[str, Tuple[int, int]]:
    """
    span_mode:
      - "line": use the full provided span (often the whole feature line).
      - "value": shrink each feature span to ONLY the value text after `value=...`
                 (supports numeric and lexical values).
    """
    if span_mode not in ("line", "value"):
        raise ValueError("span_mode must be 'line' or 'value'.")

    if span_mode == "line":
        return dict(feature_spans)

    refined: Dict[str, Tuple[int, int]] = {}
    for fid, (s, e) in feature_spans.items():
        if not (0 <= s <= e <= len(prompt)):
            raise ValueError(f"Feature span out of bounds for {fid}: {(s, e)} vs prompt_len={len(prompt)}")
        seg = prompt[s:e]
        m = _VALUE_RE.search(seg)
        if m is None:
            refined[fid] = (s, e)
            continue
        value_capture = m.group(1)
        left_trim = len(value_capture) - len(value_capture.lstrip())
        right_trim = len(value_capture.rstrip())
        vs = s + m.start(1) + left_trim
        ve = s + m.start(1) + max(left_trim, right_trim)
        if ve <= vs:
            refined[fid] = (s, e)
            continue
        refined[fid] = (vs, ve)

    return refined


def feature_token_indices_from_offsets(
    offsets: List[Tuple[int, int]],
    feature_spans: Dict[str, Tuple[int, int]],
) -> Dict[str, List[int]]:
    """
    Map character spans -> token indices via tokenizer offsets.

    Complexity is O(T * num_features) which is tiny compared to the transformer forward/backward.
    """
    feat_to_tokens: Dict[str, List[int]] = {fid: [] for fid in feature_spans.keys()}
    for ti, (ts, te) in enumerate(offsets):
        if te <= ts:
            continue
        for fid, (fs, fe) in feature_spans.items():
            if te > fs and ts < fe:
                feat_to_tokens[fid].append(ti)
    return feat_to_tokens


def _union_token_indices(feat_to_tokens: Dict[str, List[int]]) -> List[int]:
    s = set()
    for toks in feat_to_tokens.values():
        s.update(toks)
    return sorted(s)


# -----------------------------------------------------------------------------
# IG CORE HELPERS
# -----------------------------------------------------------------------------
@contextmanager
def _freeze_model_params(model):
    """
    Freeze model parameters so autograd computes gradients only w.r.t. inputs_embeds.
    This is correct for IG and often *much* faster / lower memory.
    """
    prev = [p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, req in zip(model.parameters(), prev):
            p.requires_grad_(req)


@torch.no_grad()
def _score_from_input_ids(
    model,
    prompt_ids: torch.Tensor,  # [1, T]
    labels: LabelTokens,
) -> float:
    attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
    if labels.single_token_labels:
        out = model(input_ids=prompt_ids, attention_mask=attention_mask, use_cache=False)
        logits_last = out.logits[0, -1, :]
        logprobs_last = F.log_softmax(logits_last, dim=-1)
        return float((logprobs_last[labels.case_ids[0]] - logprobs_last[labels.control_ids[0]]).item())

    logp_case = _sequence_logprob(model, prompt_ids, labels.case_ids)
    logp_control = _sequence_logprob(model, prompt_ids, labels.control_ids)
    return float((logp_case - logp_control).item())


def _sequence_logprob_from_prompt_embeds(
    model,
    prompt_embeds: torch.Tensor,  # [1, T, D] requires grad
    label_ids: List[int],
) -> torch.Tensor:
    """
    Teacher-forced log-probability of label_ids given prompt_embeds.
    Gradients flow through prompt_embeds.
    """
    device = prompt_embeds.device
    embed = model.get_input_embeddings()

    label = torch.tensor(label_ids, device=device).unsqueeze(0)  # [1, L]
    label_embeds = embed(label)  # [1, L, D]
    full_embeds = torch.cat([prompt_embeds, label_embeds], dim=1)  # [1, T+L, D]
    full_mask = torch.ones(full_embeds.shape[:2], device=device, dtype=torch.long)

    out = model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    logits = out.logits[0]  # [T+L, V]

    T = prompt_embeds.shape[1]
    L = label.shape[1]

    logp = 0.0
    for j in range(L):
        pos = T + j - 1
        lp = F.log_softmax(logits[pos, :], dim=-1)[label[0, j]]
        logp = logp + lp
    return logp


# -----------------------------------------------------------------------------
# INTEGRATED GRADIENTS
# -----------------------------------------------------------------------------
def integrated_gradients_feature_importance(
    model,
    tokenizer,
    prompt: str,
    feature_spans: Dict[str, Tuple[int, int]],
    labels: LabelTokens,
    device: str,
    steps: int = 8,
    baseline_mode: str = "mask",
    baseline_prompt: Optional[str] = None,
    span_mode: str = "value",
    check_completeness: bool = True,
    completeness_rtol: float = 5e-2,
    progress_cb: Optional[Callable[[int, int, float, float], Any]] = None,
    return_debug: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Integrated Gradients on prompt embeddings for s = logP(CASE) - logP(CONTROL).

    Why it gets slower with longer prompts:
      - If prompt is longer => more tokens T.
      - Transformer attention cost grows ~O(T^2). IG does `steps` forward+backward passes.

    span_mode:
      - "value" (recommended): attributes ONLY value text after `value=...`
      - "line": attributes entire feature line span

    baseline_mode:
      - "mask"  (recommended): baseline = same prompt, but feature VALUE tokens replaced by eos_token_id
                               (stable alignment; no baseline_prompt truncation issues)
      - "prompt": baseline embeddings from tokenizing baseline_prompt (must be semantically aligned)
      - "eos":    eos embedding repeated across all positions
      - "zero":   baseline embeddings are 0 (completeness check disabled)

    Notes:
      - Model params are frozen during IG so gradients are computed only for inputs_embeds.
      - Single-token labels: each IG step uses 1 forward pass.
      - Multi-token labels: each IG step uses 2 forward passes (CASE and CONTROL sequences).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    # Tokenize prompt with offsets so we can map spans -> token indices.
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].to(device)  # [1, T]
    offsets = enc["offset_mapping"][0].tolist()
    T = input_ids.shape[1]
    if T < 1:
        raise ValueError("Prompt tokenized to empty.")

    spans_used = _refine_feature_spans(prompt, feature_spans, span_mode=span_mode)
    feat_to_tokens = feature_token_indices_from_offsets(offsets, spans_used)
    all_feat_token_idxs = _union_token_indices(feat_to_tokens)

    embed = model.get_input_embeddings()

    # Input embeddings
    E = embed(input_ids)  # [1, T, D]

    baseline_mode = baseline_mode.lower().strip()
    baseline_ids_for_debug: Optional[torch.Tensor] = None

    if baseline_mode == "zero":
        E0 = torch.zeros_like(E)
        check_completeness = False  # delta_s vs sum(attr) isn't meaningful for all-zero embed baseline.
    elif baseline_mode == "eos":
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id; use baseline_mode='zero', 'mask', or 'prompt'.")
        eos_id = torch.tensor([[tokenizer.eos_token_id]], device=device)
        eos_emb = embed(eos_id)  # [1,1,D]
        E0 = eos_emb.repeat(1, T, 1)
        baseline_ids_for_debug = torch.full_like(input_ids, tokenizer.eos_token_id)
    elif baseline_mode == "prompt":
        if baseline_prompt is None:
            raise ValueError("baseline_mode='prompt' requires baseline_prompt=str.")
        enc0 = tokenizer(baseline_prompt, return_tensors="pt", add_special_tokens=False)
        ids0 = enc0["input_ids"].to(device)

        # Length-match baseline ids to prompt length (warning: alignment is only safe if templates tokenize similarly).
        if ids0.shape[1] < T:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no eos_token_id to pad baseline_prompt; use baseline_mode='mask' or 'zero'.")
            pad = torch.full((1, T - ids0.shape[1]), tokenizer.eos_token_id, device=device, dtype=ids0.dtype)
            ids0 = torch.cat([ids0, pad], dim=1)
            warnings.warn(
                f"[IG] baseline_prompt shorter than prompt; padded baseline ids to length {T} with eos_token_id.",
                RuntimeWarning,
            )
        elif ids0.shape[1] > T:
            ids0 = ids0[:, :T]
            warnings.warn(
                f"[IG] baseline_prompt longer than prompt; truncated baseline ids to length {T}. "
                f"If you see this often, prefer baseline_mode='mask' for exact alignment.",
                RuntimeWarning,
            )

        E0 = embed(ids0)
        baseline_ids_for_debug = ids0
    elif baseline_mode == "mask":
        # Best alignment: keep the prompt identical, only mask the VALUE tokens.
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id; use baseline_mode='zero' or 'prompt'.")
        ids0 = input_ids.clone()
        if all_feat_token_idxs:
            ids0[0, all_feat_token_idxs] = tokenizer.eos_token_id
        else:
            warnings.warn(
                "[IG] baseline_mode='mask' but no feature tokens were found from spans. "
                "Attribution will be ~0. Check your feature_spans/span_mode.",
                RuntimeWarning,
            )
        E0 = embed(ids0)
        baseline_ids_for_debug = ids0
    else:
        raise ValueError("baseline_mode must be one of: 'mask', 'prompt', 'eos', 'zero'.")

    # Integrated gradients path
    delta = E - E0
    grad_sum = torch.zeros_like(E)

    t0 = time.perf_counter()

    # Freeze params: we only need gradients w.r.t. E_alpha
    with _freeze_model_params(model):
        for j in range(1, steps + 1):
            alpha = j / steps

            # Build interpolated embeddings; leaf tensor for autograd.
            E_alpha = (E0 + alpha * delta).detach()
            E_alpha.requires_grad_(True)

            attention_mask = torch.ones((1, T), device=device, dtype=torch.long)

            if labels.single_token_labels:
                out = model(inputs_embeds=E_alpha, attention_mask=attention_mask, use_cache=False)
                logits_last = out.logits[:, -1, :]  # [1, V]
                logprobs_last = F.log_softmax(logits_last, dim=-1)
                s = logprobs_last[0, labels.case_ids[0]] - logprobs_last[0, labels.control_ids[0]]
            else:
                logp_case = _sequence_logprob_from_prompt_embeds(model, E_alpha, labels.case_ids)
                logp_control = _sequence_logprob_from_prompt_embeds(model, E_alpha, labels.control_ids)
                s = logp_case - logp_control

            # Compute gradient only w.r.t. E_alpha (no param grads)
            grad = torch.autograd.grad(
                outputs=s,
                inputs=E_alpha,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            if grad is None:
                raise RuntimeError("IG: grad is None. Check that inputs_embeds participates in the graph.")

            if torch.isnan(grad).any() or torch.isinf(grad).any():
                raise RuntimeError("IG: NaN/Inf in gradients. Try fewer steps, different baseline, or CPU/CUDA.")

            grad_sum += grad.detach()

            if progress_cb is not None:
                elapsed = time.perf_counter() - t0
                rate = elapsed / j
                eta = rate * (steps - j)
                progress_cb(j, steps, elapsed, eta)

    avg_grad = grad_sum / steps
    IG = delta * avg_grad  # [1, T, D]

    # Token-level attributions
    token_scores = IG.sum(dim=-1).squeeze(0)  # [T]

    # Feature-level aggregation
    feat_scores: Dict[str, float] = {}
    for fid, tok_idxs in feat_to_tokens.items():
        feat_scores[fid] = float(token_scores[tok_idxs].sum().item()) if tok_idxs else 0.0

    debug: Dict[str, float] = {
        "steps": float(steps),
        "T_tokens": float(T),
        "n_features": float(len(feature_spans)),
        "n_feature_tokens_union": float(len(all_feat_token_idxs)),
        "span_mode_line/value": 0.0 if span_mode == "line" else 1.0,
        "baseline_mode_code": float({"zero": 0, "eos": 1, "prompt": 2, "mask": 3}.get(baseline_mode, -1)),
        "len_case_ids": float(len(labels.case_ids or [])),
        "len_control_ids": float(len(labels.control_ids or [])),
    }

    # Completeness check: sum(token attrs) ≈ s(x) - s(x0) for token-id baselines.
    if check_completeness and baseline_mode in ("eos", "prompt", "mask"):
        if baseline_ids_for_debug is None:
            warnings.warn("[IG] completeness check skipped: baseline_ids_for_debug is None.", RuntimeWarning)
        else:
            with torch.no_grad():
                s_x = _score_from_input_ids(model, input_ids, labels)
                s_x0 = _score_from_input_ids(model, baseline_ids_for_debug, labels)
                delta_s = s_x - s_x0
                total_attr = float(token_scores.sum().item())
                abs_err = abs(total_attr - delta_s)

                debug.update(
                    {
                        "s_x": float(s_x),
                        "s_x0": float(s_x0),
                        "delta_s": float(delta_s),
                        "total_token_attr": float(total_attr),
                        "completeness_abs_err": float(abs_err),
                    }
                )

                denom = abs(delta_s) + 1e-6
                if abs_err > completeness_rtol * denom:
                    warnings.warn(
                        f"[IG] Completeness check failed beyond tolerance: "
                        f"sum(attr)={total_attr:+.6f} vs delta_s={delta_s:+.6f} "
                        f"(abs_err={abs_err:.6f}, rtol={completeness_rtol}).",
                        RuntimeWarning,
                    )

    if return_debug:
        return feat_scores, debug

    return feat_scores
