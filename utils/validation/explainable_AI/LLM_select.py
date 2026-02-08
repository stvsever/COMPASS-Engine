# LLM_select.py
from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# OpenAI Python SDK (new-style client)
from openai import OpenAI


class LLMSelectPayload(BaseModel):
    """
    Structured output schema:
      - leaf_scores:  dict leaf_id -> int in [1, 1000]
      - parent_scores: dict parent_id -> int in [1, 1000]
    """
    leaf_scores: Dict[str, int] = Field(default_factory=dict)
    parent_scores: Dict[str, int] = Field(default_factory=dict)


def _l1_normalize(scores: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    s = sum(abs(v) for v in scores.values()) + eps
    return {k: abs(float(v)) / s for k, v in scores.items()}


def _validate_expected_keys(name: str, got: Dict[str, int], expected: List[str]) -> None:
    got_keys = set(got.keys())
    exp_keys = set(expected)
    missing = sorted(exp_keys - got_keys)
    extra = sorted(got_keys - exp_keys)
    if missing or extra:
        raise ValueError(
            f"{name} keys mismatch. missing={missing[:5]}{'...' if len(missing)>5 else ''} "
            f"extra={extra[:5]}{'...' if len(extra)>5 else ''}"
        )


def _validate_range(name: str, got: Dict[str, int], lo: int = 1, hi: int = 1000) -> None:
    bad = [(k, v) for k, v in got.items() if not isinstance(v, int) or v < lo or v > hi]
    if bad:
        k, v = bad[0]
        raise ValueError(f"{name} score out of range or not int: {k}={v} (expected int in [{lo},{hi}])")


def _build_prompt(phenotype: str, leaf_ids: List[str], parent_ids: List[str]) -> Tuple[str, str]:
    # IMPORTANT: Include the word "json" in the messages when using JSON output mode.
    system = (
        "You are a scientist expert helping with feature selection.\n"
        "You will receive a task description and lists of feature names.\n"
        "Return integer importance scores from 1 to 1000 for every feature.\n"
        "Higher = more important for predicting the phenotype.\n"
        "Use general domain knowledge; do not invent new features.\n\n"
        "Output must be valid json and follow exactly this schema:\n"
        "{\n"
        '  "leaf_scores": { "<leaf_id>": <int 1..1000>, ... },\n'
        '  "parent_scores": { "<parent_id>": <int 1..1000>, ... }\n'
        "}\n"
        "Return ONLY the json object. No prose, no markdown."
    )

    user = (
        f"Task: Predict whether a subject is '{phenotype} CASE' vs '{phenotype} CONTROL' from phenotypic features.\n\n"
        "Rate importance for:\n"
        "1) LEAF features (more specific)\n"
        "2) PARENT features (coarser groups)\n\n"
        "Rules:\n"
        "- Every listed feature must have a score.\n"
        "- Scores must be integers 1..1000.\n"
        "- Distribute scores realistically: only a few features should be very high.\n\n"
        f"LEAF_FEATURES ({len(leaf_ids)}):\n"
        + "\n".join(f"- {x}" for x in leaf_ids)
        + f"\n\nPARENT_FEATURES ({len(parent_ids)}):\n"
        + "\n".join(f"- {x}" for x in parent_ids)
        + "\n\nRemember: return ONLY a json object with keys leaf_scores and parent_scores."
    )
    return system, user


def _extract_output_text(resp: Any) -> Optional[str]:
    """
    Try to extract the model's text from an OpenAI Responses API result across SDK variants.
    """
    # Most common in the new SDK
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # Try structured output container variants
    out = getattr(resp, "output", None)
    if isinstance(out, list):
        chunks: List[str] = []
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    # Typical: {"type":"output_text","text":"..."}
                    ctype = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)
                    ctext = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                    if ctype in ("output_text", "text") and isinstance(ctext, str):
                        chunks.append(ctext)
        joined = "".join(chunks).strip()
        return joined if joined else None

    return None


def _call_openai_structured(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_retries: int = 3,
    retry_sleep_s: float = 1.0,
) -> LLMSelectPayload:
    """
    Preferred:
      - responses.parse(..., text_format=LLMSelectPayload) if supported by the model.

    Fallback:
      - responses.create(..., text={"format":{"type":"json_object"}}) + Pydantic validate

    NOTE: JSON mode requires that the prompt contains the word "json" somewhere.
    We bake that into _build_prompt().
    """
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        # 1) Try Structured Outputs (Pydantic)
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=LLMSelectPayload,
                #temperature=temperature,
            )
            payload = getattr(resp, "output_parsed", None)
            if not isinstance(payload, LLMSelectPayload):
                raise TypeError("responses.parse did not return output_parsed as LLMSelectPayload.")
            return payload
        except Exception as e1:
            last_err = e1

        # 2) Fallback: JSON object mode + validate
        try:
            resp2 = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                #temperature=temperature,
                text={"format": {"type": "json_object"}},
            )
            txt = _extract_output_text(resp2)
            if not txt:
                raise RuntimeError("OpenAI SDK response missing extractable output text in JSON mode fallback.")
            return LLMSelectPayload.model_validate_json(txt)
        except Exception as e2:
            last_err = e2
            if attempt < max_retries:
                time.sleep(retry_sleep_s)

    assert last_err is not None
    raise last_err


def get_llm_select_scores(
    phenotype: str,
    leaf_ids: List[str],
    parent_ids: List[str],
    repeats: int = 3, #NOTE: for testing ; change to higher number during actual run
    model: str = "gpt-5-nano", # NOTE: for testing ; change to better models during actual run
    temperature: float = 0.7,
    max_retries: int = 3,
    client_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Dict[str, float]]], List[Dict[str, Dict[str, float]]]]:
    """
    Returns:
      leaf_runs:   list length=repeats, each item {"raw": {leaf->int}, "norm": {leaf->float}}
      parent_runs: list length=repeats, each item {"raw": {parent->int}, "norm": {parent->float}}

    Normalization is L1 over ABS values, performed separately for leaves and parents.
    """
    load_dotenv()  # reads OPENAI_API_KEY from .env if present
    kwargs = dict(client_kwargs or {})
    client = OpenAI(**kwargs) if kwargs else OpenAI()

    system, user = _build_prompt(phenotype=phenotype, leaf_ids=leaf_ids, parent_ids=parent_ids)

    leaf_runs: List[Dict[str, Dict[str, float]]] = []
    parent_runs: List[Dict[str, Dict[str, float]]] = []

    for _ in range(repeats):
        payload = _call_openai_structured(
            client=client,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_retries=max_retries,
        )

        leaf_raw: Dict[str, int] = payload.leaf_scores
        parent_raw: Dict[str, int] = payload.parent_scores

        # Validate
        _validate_expected_keys("leaf_scores", leaf_raw, leaf_ids)
        _validate_expected_keys("parent_scores", parent_raw, parent_ids)
        _validate_range("leaf_scores", leaf_raw, 1, 1000)
        _validate_range("parent_scores", parent_raw, 1, 1000)

        leaf_norm = _l1_normalize({k: float(v) for k, v in leaf_raw.items()})
        parent_norm = _l1_normalize({k: float(v) for k, v in parent_raw.items()})

        # Keep raw as float for compatibility with downstream code,
        # but they will still be integer-valued
        leaf_runs.append({"raw": {k: float(v) for k, v in leaf_raw.items()}, "norm": leaf_norm})
        parent_runs.append({"raw": {k: float(v) for k, v in parent_raw.items()}, "norm": parent_norm})

    return leaf_runs, parent_runs
