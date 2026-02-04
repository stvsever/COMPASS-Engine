""""
COMPASS JSON Parser

Robust JSON extraction from LLM responses with error recovery.
"""

import re
import json
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("compass.json_parser")


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain markdown or other formatting.
    
    Handles:
    - Pure JSON
    - JSON in ```json code blocks
    - JSON with leading/trailing text
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Try pure JSON first
    if text.startswith('{') or text.startswith('['):
        return text
    
    # Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    
    for match in matches:
        match = match.strip()
        if match.startswith('{') or match.startswith('['):
            return match
    
    # Try to find JSON object anywhere in text
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, text)
    if match:
        return match.group()
    
    # Try to find JSON array
    json_array_pattern = r'\[[\s\S]*\]'
    match = re.search(json_array_pattern, text)
    if match:
        return match.group()
    
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
    
    json_str = extract_json_from_text(response_text)
    
    if not json_str:
        logger.warning("No JSON found in response")
        if default:
            return default
        raise ValueError("No JSON found in LLM response")
    
    try:
        parsed = json.loads(json_str)
        
        # Validate expected keys if provided
        if expected_keys:
            missing_keys = [k for k in expected_keys if k not in parsed]
            if missing_keys:
                logger.warning(f"Missing expected keys: {missing_keys}")
        
        return parsed
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        
        # Try to fix common issues
        fixed = try_fix_json(json_str)
        if fixed:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        
        if default:
            return default
        raise ValueError(f"Invalid JSON in LLM response: {e}")


def try_fix_json(json_str: str) -> Optional[str]:
    """
    Attempt to fix common JSON issues.
    """
    # Remove trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
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
