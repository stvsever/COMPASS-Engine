"""
Pre-compute embeddings for all unique hierarchical features in the dataset.
Optimized with ThreadPoolExecutor for high concurrency (200 workers).
Self-contained imports for standalone execution.

Logic:
1. Scan all participant directories for `multimodal_data.json`.
2. Recursively parse the JSON to extract leaf nodes.
3. Construct hierarchical feature strings: "feature <- parent1 <- parent2 <- parent3".
4. Deduplicate strings across the entire dataset.
5. Generate embeddings using OpenAI (text-embedding-3-large) and cache them.
6. Report statistics.
"""

import os
import sys
import json
import hashlib
import pickle
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Import Robustness ---
# Add project root to path if running as script from utils/
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import get_settings
    # We can try to use the centralized client or just direct OpenAI for simplicity logic here
    # to avoid the circular dependency hell we saw earlier.
    # The user wants "joint functions", so let's try to be consistent,
    # but direct use of OpenAI library here guarantees standalone stability.
except ImportError:
    pass

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path("/Users/stijnvanseveren/PythonProjects/IIS_BIOBIZKAIA/INFERENCE_PIPELINE/data/__FEATURES__/COMPASS_data")
# Handle running from root or utils
if Path(".cache").exists():
    CACHE_DIR = Path(".cache/embeddings")
else:
    # Assume we might be in utils or root, aim for relative to project root
    # Best guess: ../.cache/embeddings if running from utils, or .cache/embeddings if root
    # We'll default to absolute path based on file location to be safe
    CACHE_DIR = Path(__file__).parent.parent / ".cache" / "embeddings"

EMBEDDING_MODEL = "text-embedding-3-large"
MAX_WORKERS = 200
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("Error: OPENAI_API_KEY not found in environment.")
    # Attempt to load from settings if env fail
    try:
        from config.settings import get_settings
        settings = get_settings()
        API_KEY = settings.openai_api_key
    except:
        pass

if not API_KEY:
    print("CRITICAL: Could not find OpenAI API Key.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

# Thread-safe counters
stats_lock = threading.Lock()
stats = {
    "processed": 0,
    "skipped": 0,
    "errors": 0,
    "retries": 0
}

@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception)
)
def get_embedding_safe(text: str) -> List[float]:
    """Get embedding with retry logic."""
    try:
        # Pre-truncation (naive)
        if len(text) > 30000:
            text = text[:30000]
            
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        # Check for rate limit specifically to log it?
        # print(f"Retry needed for '{text[:20]}...': {e}")
        raise

def process_single_embedding(text: str) -> str:
    """Worker function to process a single text string."""
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    cache_file = CACHE_DIR / f"{text_hash}.pkl"
    
    if cache_file.exists():
        with stats_lock:
            stats["skipped"] += 1
        return "skipped"
        
    try:
        emb = get_embedding_safe(text)
        
        # Atomic write pattern not strictly needed for cache (idempotent), but good practice
        # Direct dump is fine for now
        with open(cache_file, "wb") as f:
            pickle.dump(emb, f)
            
        with stats_lock:
            stats["processed"] += 1
        return "processed"
        
    except Exception as e:
        with stats_lock:
            stats["errors"] += 1
        return f"error: {str(e)}"

def get_hierarchical_strings(data: Dict[str, Any], parents: List[str] = None) -> Set[str]:
    """Recursively extract feature strings."""
    if parents is None:
        parents = []
        
    strings = set()
    
    if isinstance(data, dict):
        if "_leaves" in data:
            for leaf in data["_leaves"]:
                if isinstance(leaf, dict) and "feature" in leaf:
                    # feature <- p1 <- p2 <- p3
                    context_parents = parents[-3:][::-1]
                    parts = [leaf["feature"]]
                    parts.extend(context_parents)
                    feature_str = " <- ".join(parts)
                    strings.add(feature_str)
        
        for key, value in data.items():
            if key != "_leaves":
                new_parents = parents + [key]
                strings.update(get_hierarchical_strings(value, new_parents))
                
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "feature" in item:
                 context_parents = parents[-3:][::-1]
                 parts = [item["feature"]]
                 parts.extend(context_parents)
                 feature_str = " <- ".join(parts)
                 strings.add(feature_str)
            else:
                strings.update(get_hierarchical_strings(item, parents))

    return strings

def main():
    print("="*60)
    print("COMPASS Feature Embedding Pre-computation (Optimized)")
    print("="*60)
    print(f"Concurrency: {MAX_WORKERS} threads")
    print(f"Data Dir:    {DATA_DIR}")
    print(f"Cache Dir:   {CACHE_DIR}")
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Scanning
    print("\nScanning for data files...")
    files = list(DATA_DIR.rglob("multimodal_data.json"))
    print(f"Found {len(files)} multimodal data files.")
    
    all_unique_strings = set()
    demographic_strings = set() # Track for cleanup
    
    print("Extracting hierarchical feature strings (Excluding DEMOGRAPHICS)...")
    
    for f in tqdm(files, desc="Parsing Files"):
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
            for domain, content in data.items():
                # EXCLUSION LOGIC
                if domain == "DEMOGRAPHICS":
                    # Collect these for cleanup, but do NOT add to all_unique_strings
                    demographic_strings.update(get_hierarchical_strings(content, [domain]))
                    continue
                    
                all_unique_strings.update(get_hierarchical_strings(content, [domain]))
        except Exception as e:
            pass # Ignore read errors

    total_strings = len(all_unique_strings)
    
    # --- CLEANUP PHASE ---
    if demographic_strings:
        print(f"\nPerforming DEMOGRAPHICS Cleanup ({len(demographic_strings)} variants found)...")
        cleaned_count = 0
        for text in demographic_strings:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            cache_file = CACHE_DIR / f"{text_hash}.pkl"
            if cache_file.exists():
                try:
                    os.remove(cache_file)
                    cleaned_count += 1
                except OSError:
                    pass
        print(f"Removed {cleaned_count} cached demographic embeddings.")
        
    print(f"\nTotal Valid Unique Feature Strings: {total_strings}")

    # 2. Embedding
    print(f"\nGeneratring embeddings with {MAX_WORKERS} workers...")
    
    sorted_strings = sorted(list(all_unique_strings))
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_embedding, text) for text in sorted_strings]
        
        # Monitor with tqdm
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            pass

    duration = time.time() - start_time
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Time Taken:     {duration:.1f} seconds")
    print(f"Total Features: {total_strings}")
    print(f"New Embeddings: {stats['processed']}")
    print(f"Cached/Skipped: {stats['skipped']}")
    print(f"Errors:         {stats['errors']}")
    print("="*60)

if __name__ == "__main__":
    main()
