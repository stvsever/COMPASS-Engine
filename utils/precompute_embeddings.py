"""
Pre-compute global feature-path embeddings into the SQLite embedding store.

This script populates reusable path embeddings of the form:
    feature <- parent1 <- parent2 <- parent3
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from openai import OpenAI

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))

from config.settings import get_settings, LLMBackend
from utils.core.embedding_store import get_embedding_store

load_dotenv()


def _feature_strings(data: Any, parents: List[str] | None = None) -> Set[str]:
    parents = parents or []
    out: Set[str] = set()

    if isinstance(data, dict):
        leaves = data.get("_leaves")
        if isinstance(leaves, list):
            for leaf in leaves:
                if not isinstance(leaf, dict):
                    continue
                name = leaf.get("feature") or leaf.get("field_name")
                if not name:
                    continue
                path = [str(p) for p in parents if str(p).strip()]
                out.add(" <- ".join([str(name), *path[-3:][::-1]]))
        for key, value in data.items():
            if key == "_leaves":
                continue
            if isinstance(value, (dict, list)):
                out |= _feature_strings(value, parents + [str(key)])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "feature" in item or "field_name" in item:
                    name = item.get("feature") or item.get("field_name")
                    path = item.get("path_in_hierarchy") or []
                    if not isinstance(path, list):
                        path = []
                    out.add(" <- ".join([str(name), *[str(p) for p in path[-3:][::-1]]]))
                else:
                    out |= _feature_strings(item, parents)
    return out


def main() -> None:
    settings = get_settings()
    if settings.models.backend == LLMBackend.OPENROUTER:
        api_key = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found.")
        client = OpenAI(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                k: v
                for k, v in {
                    "HTTP-Referer": settings.openrouter_site_url,
                    "X-Title": settings.openrouter_app_name,
                }.items()
                if v
            } or None,
        )
    else:
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found.")
        client = OpenAI(api_key=api_key)
    store = get_embedding_store()
    model = settings.models.embedding_model or "text-embedding-3-large"
    if settings.models.backend == LLMBackend.OPENROUTER and "/" not in model:
        if model.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-")):
            model = f"openai/{model}"

    data_root = settings.paths.base_dir.parent / "data" / "__FEATURES__" / "COMPASS_data"
    files = list(data_root.rglob("multimodal_data.json"))
    if not files:
        print(f"No multimodal_data.json files found under {data_root}")
        return

    all_strings: Set[str] = set()
    for path in files:
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            for domain, content in payload.items():
                all_strings |= _feature_strings(content, [str(domain)])
        except Exception:
            continue

    strings = sorted(s for s in all_strings if s.strip())
    if not strings:
        print("No feature-path strings extracted.")
        return

    try:
        max_workers = int(os.getenv("COMPASS_EMBEDDING_MAX_WORKERS", "200"))
    except Exception:
        max_workers = 200
    workers = max(1, min(max_workers, len(strings)))

    def embed_fn(text: str, embed_model: str):
        response = client.embeddings.create(input=text, model=embed_model)
        return response.data[0].embedding

    print(f"Embedding {len(strings)} path strings with {workers} workers...")
    completed = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                store.get_or_create_global,
                text,
                model,
                embed_fn,
                "feature_path",
            )
            for text in strings
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
                completed += 1
            except Exception:
                errors += 1

    print(f"Done. completed={completed} errors={errors} db={store.db_path}")


if __name__ == "__main__":
    main()
