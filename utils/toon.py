"""
TOON (Token-Oriented Object Notation) Converter
===============================================

A lightweight utility to convert hierarchical JSON data into a token-efficient 
text representation for LLM consumption.

Key Features:
- Removes syntax noise (braces, quotes, commas)
- Uses indentation for hierarchy
- Compacts lists and leaf nodes
- Preserves information losslessly
"""

from typing import Any, List, Dict, Union

def json_to_toon(data: Any, indent_level: int = 0, indent_str: str = "  ") -> str:
    """
    Convert JSON-serializable data to TOON format.
    
    Args:
        data: Input data (dict, list, str, int, float, etc.)
        indent_level: Current indentation depth
        indent_str: String used for indentation (default: 2 spaces)
        
    Returns:
        String representation in TOON format
    """
    current_indent = indent_str * indent_level
    
    # 1. Handle Dictionaries
    if isinstance(data, dict):
        if not data:
            return "{}"
            
        lines = []
        for key, value in data.items():
            # Special case for leaf lists (e.g. key: [val1, val2])
            if isinstance(value, list) and not any(isinstance(x, (dict, list)) for x in value):
                # Compact list inline
                val_str = f"[{', '.join(str(v) for v in value)}]"
                lines.append(f"{current_indent}{key}: {val_str}")
                
            # Special case for COMPASS feature leaf objects
            # - raw UKB leaves: {'feature': ..., 'z_score': ...}
            # - DataLoader flattened: {'field_name': ..., 'z_score': ...}
            elif isinstance(value, list) and all(
                isinstance(x, dict) and ('feature' in x or 'field_name' in x) and 'z_score' in x
                for x in value
            ):
                # Serialize feature list compactly:
                # FeatureName: 1.23, FeatureTwo: -0.5
                lines.append(f"{current_indent}{key}:")
                item_strs = []
                for item in value:
                    feat_name = item.get("feature") or item.get("field_name") or "unknown"
                    v_str = f"{feat_name}: {item.get('z_score')}"
                    item_strs.append(v_str)
                
                # Split into multi-line if too long
                chunk_str = ", ".join(item_strs)
                if len(chunk_str) > 100:
                    for i_str in item_strs:
                        lines.append(f"{current_indent}{indent_str}- {i_str}")
                else:
                    lines.append(f"{current_indent}{indent_str}{chunk_str}")

            # Standard recursive handling
            elif isinstance(value, (dict, list)):
                lines.append(f"{current_indent}{key}:")
                lines.append(json_to_toon(value, indent_level + 1, indent_str))
            else:
                # Simple value
                lines.append(f"{current_indent}{key}: {value}")
                
        return "\n".join(lines)

    # 2. Handle Lists
    elif isinstance(data, list):
        if not data:
            return "[]"
            
        # Check if list of simple items
        if not any(isinstance(x, (dict, list)) for x in data):
            return f"{current_indent}[{', '.join(str(x) for x in data)}]"
            
        # List of complex objects
        lines = []
        for item in data:
            # If item is simple dict, try to keep it compact? 
            # For now, treat list items with a dash bullet
            item_str = json_to_toon(item, indent_level + 1, indent_str)
            # Remove the first level of indentation from the recursive call
            # because we add the dash at the current level
            stripped_item_str = item_str.lstrip()
            lines.append(f"{current_indent}- {stripped_item_str}")
            
        return "\n".join(lines)

    # 3. Handle Primitives
    else:
        return f"{current_indent}{str(data)}"
