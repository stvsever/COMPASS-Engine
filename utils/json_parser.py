"""
COMPASS JSON Parser

Robust JSON extraction from LLM responses with error recovery.
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("compass.json_parser")


def _balanced_json_spans(text: str) -> List[Tuple[int, int]]:
    """
    Find candidate JSON spans using brace/bracket balancing with string awareness.
    """
    spans: List[Tuple[int, int]] = []
    if not text:
        return spans

    stack: List[str] = []
    start_idx: Optional[int] = None
    in_str = False
    escaped = False

    for idx, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue

        if ch in "{[":
            if not stack:
                start_idx = idx
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                continue
            opener = stack[-1]
            if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                stack.pop()
                if not stack and start_idx is not None:
                    spans.append((start_idx, idx + 1))
                    start_idx = None
            else:
                # Corrupted nesting; reset scan state.
                stack = []
                start_idx = None

    return spans


def _extract_json_candidates(text: str) -> List[str]:
    """
    Collect JSON candidates from raw output in priority order.
    """
    if not text:
        return []

    candidates: List[str] = []
    seen: set[str] = set()

    def _add(value: Optional[str]) -> None:
        candidate = str(value or "").strip()
        if not candidate:
            return
        if not (candidate.startswith("{") or candidate.startswith("[")):
            return
        if candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    stripped = text.strip()
    _add(stripped)

    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    for match in re.findall(code_block_pattern, text):
        _add(match)

    for start, end in _balanced_json_spans(text):
        _add(text[start:end])
        if len(candidates) >= 40:
            break

    # Last-resort non-greedy regex fallbacks.
    for pattern in (r"\{[\s\S]*?\}", r"\[[\s\S]*?\]"):
        for match in re.finditer(pattern, text):
            _add(match.group())
            if len(candidates) >= 60:
                break
        if len(candidates) >= 60:
            break

    return candidates


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Return the highest-priority JSON candidate from free-form text.
    """
    candidates = _extract_json_candidates(text)
    if candidates:
        return candidates[0]
    return None


def parse_json_response(
    response_text: str,
    expected_keys: Optional[list] = None,
    default: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Parse JSON from LLM response with error handling.
    
    Args:
        response_text: Raw text from LLM
        expected_keys: Optional list of keys that should be present
        default: Default value if parsing fails
    
    Returns:
        Parsed JSON as dictionary
    
    Raises:
        ValueError: If JSON cannot be parsed and no default provided
    """
    if default is None:
        default = {}
    
    # Pre-processing: Remove <think>...</think> blocks common in reasoning models
    response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text, flags=re.IGNORECASE)
    
    candidates = _extract_json_candidates(response_text)

    if not candidates:
        logger.warning("No JSON found in response")
        if default:
            return default
        raise ValueError("No JSON found in LLM response")

    parse_errors: List[str] = []
    parsed_candidates: List[Tuple[Tuple[int, int, int], int, Any]] = []

    def _candidate_score(parsed: Any) -> Tuple[int, int, int]:
        """
        Rank parsed JSON candidates.
        Priority:
        1) Prefer dictionary top-level payloads over arrays
        2) Number of expected keys present (when provided)
        3) Structural richness (dict/list length)
        4) Prefer later candidates in the output (often the final answer)
        """
        object_preference = 0
        expected_hit = 0
        richness = 0
        if isinstance(parsed, dict):
            object_preference = 1
            richness = len(parsed)
            if expected_keys:
                expected_hit = sum(1 for k in expected_keys if k in parsed)
        elif isinstance(parsed, list):
            richness = len(parsed)
            if expected_keys:
                # When a dict schema is expected, deprioritize list candidates.
                expected_hit = -1
        return (object_preference, expected_hit, richness)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            parsed_candidates.append((_candidate_score(parsed), len(parsed_candidates), parsed))
            continue
        except json.JSONDecodeError as e:
            parse_errors.append(str(e))
            fixed = try_fix_json(candidate)
            if fixed:
                try:
                    parsed = json.loads(fixed)
                    parsed_candidates.append((_candidate_score(parsed), len(parsed_candidates), parsed))
                    continue
                except json.JSONDecodeError as fix_err:
                    parse_errors.append(str(fix_err))

    if parsed_candidates:
        # Tie-break on candidate order (later is preferred).
        best = max(parsed_candidates, key=lambda item: (item[0][0], item[0][1], item[0][2], item[1]))
        parsed = best[2]
        if expected_keys and isinstance(parsed, dict):
            missing_keys = [k for k in expected_keys if k not in parsed]
            if missing_keys:
                logger.warning(f"Missing expected keys: {missing_keys}")
        return parsed

    if parse_errors:
        logger.error("JSON parsing failed across %s candidates", len(candidates))
    if default:
        return default
    detail = parse_errors[0] if parse_errors else "unknown decode error"
    raise ValueError(f"Invalid JSON in LLM response: {detail}")


def try_fix_json(json_str: str) -> Optional[str]:
    """
    Attempt to fix common JSON issues.
    """
    def escape_unescaped_newlines(text: str) -> str:
        out = []
        in_str = False
        escaped = False
        for ch in text:
            if in_str:
                if escaped:
                    escaped = False
                    out.append(ch)
                    continue
                if ch == "\\":
                    escaped = True
                    out.append(ch)
                    continue
                if ch == "\"":
                    in_str = False
                    out.append(ch)
                    continue
                if ch == "\n":
                    out.append("\\n")
                    continue
                if ch == "\r":
                    continue
                if ch == "\t":
                    out.append("\\t")
                    continue
                out.append(ch)
                continue
            if ch == "\"":
                in_str = True
            out.append(ch)
        return "".join(out)

    def quote_unquoted_keys(text: str) -> str:
        out = []
        i = 0
        in_str = False
        escaped = False
        n = len(text)
        while i < n:
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                    out.append(ch)
                    i += 1
                    continue
                if ch == "\\":
                    escaped = True
                    out.append(ch)
                    i += 1
                    continue
                if ch == "\"":
                    in_str = False
                out.append(ch)
                i += 1
                continue
            if ch == "\"":
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch.isalpha() or ch == "_":
                j = i
                while j < n and (text[j].isalnum() or text[j] in "_-"):
                    j += 1
                k = j
                while k < n and text[k].isspace():
                    k += 1
                if k < n and text[k] == ":":
                    key = text[i:j]
                    out.append(f"\"{key}\"")
                    out.append(text[j:k])
                    out.append(":")
                    i = k + 1
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    # Remove trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # Escape unescaped newlines/tabs inside strings
    fixed = escape_unescaped_newlines(fixed)

    # Quote unquoted keys (JSON5/JS-style -> JSON)
    fixed = quote_unquoted_keys(fixed)
    
    # Fix single quotes
    if "'" in fixed and '"' not in fixed:
        fixed = fixed.replace("'", '"')
    
    # Try to balance braces
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    if open_braces > close_braces:
        fixed = fixed + ('}' * (open_braces - close_braces))
    
    open_brackets = fixed.count('[')
    close_brackets = fixed.count(']')
    if open_brackets > close_brackets:
        fixed = fixed + (']' * (open_brackets - close_brackets))
    
    return fixed


def safe_get(
    data: Dict[str, Any],
    *keys: str,
    default: Any = None
) -> Any:
    """
    Safely get nested value from dictionary.
    
    Usage:
        value = safe_get(data, "level1", "level2", "key", default="fallback")
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def validate_json_structure(
    data: Dict[str, Any],
    schema: Dict[str, type]
) -> Dict[str, list]:
    """
    Validate JSON structure against expected schema.
    
    Args:
        data: JSON data to validate
        schema: Dict mapping key names to expected types
    
    Returns:
        Dict with 'missing' and 'wrong_type' lists
    """
    issues = {"missing": [], "wrong_type": []}
    
    for key, expected_type in schema.items():
        if key not in data:
            issues["missing"].append(key)
        elif not isinstance(data[key], expected_type):
            issues["wrong_type"].append(
                f"{key}: expected {expected_type.__name__}, got {type(data[key]).__name__}"
            )
    
    return issues


def json_to_markdown(data: Dict[str, Any], indent: int = 0) -> str:
    """
    Convert JSON structure to readable markdown.
    
    Useful for logging and debugging.
    """
    lines = []
    prefix = "  " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}**{key}**:")
            lines.append(json_to_markdown(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}**{key}**:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  - {json.dumps(item)[:100]}...")
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}**{key}**: {value}")
    
    return "\n".join(lines)
