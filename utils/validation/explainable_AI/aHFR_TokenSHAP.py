# aHFR_TokenSHAP.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any, Union
from collections import deque
import random
import math
import time


# -----------------------
# TIME / ETA
# -----------------------
def _eta(elapsed: float, done: int, total: int) -> float:
    if done <= 0:
        return float("inf")
    rate = elapsed / done
    return rate * max(total - done, 0)


# -----------------------
# HIERARCHY CORE
# -----------------------
def infer_root(children: Dict[str, List[str]]) -> str:
    """Infer a single root for a rooted tree represented as adjacency lists."""
    parents = set(children.keys())
    all_children: Set[str] = set()
    for chs in children.values():
        all_children.update(chs)

    roots = list(parents - all_children)
    if len(roots) != 1:
        raise ValueError(
            f"Cannot infer unique root. Root candidates={roots}. "
            f"Provide root explicitly or fix hierarchy."
        )
    return roots[0]


@dataclass(frozen=True)
class Hierarchy:
    """
    A rooted tree hierarchy.
    - children: adjacency mapping parent -> list(children)
    - root: root id
    """
    children: Dict[str, List[str]]
    root: str

    parent_of: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    depth: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        parent_of: Dict[str, str] = {}
        nodes: Set[str] = set()

        nodes.add(self.root)
        for p, chs in self.children.items():
            nodes.add(p)
            for c in chs:
                nodes.add(c)

        for p, chs in self.children.items():
            for c in chs:
                if c in parent_of:
                    raise ValueError(
                        f"Hierarchy is not a tree: node '{c}' has multiple parents "
                        f"('{parent_of[c]}' and '{p}')."
                    )
                parent_of[c] = p

        seen: Set[str] = set()
        stack: Set[str] = set()

        def dfs(u: str):
            if u in stack:
                raise ValueError(f"Hierarchy has a cycle involving '{u}'.")
            if u in seen:
                return
            seen.add(u)
            stack.add(u)
            for v in self.children.get(u, []):
                dfs(v)
            stack.remove(u)

        dfs(self.root)

        unreachable = nodes - seen
        if unreachable:
            raise ValueError(
                f"Hierarchy contains nodes not reachable from root '{self.root}': "
                f"{sorted(unreachable)}"
            )

        # compute depth (BFS)
        depth: Dict[str, int] = {self.root: 0}
        q = deque([self.root])
        while q:
            u = q.popleft()
            du = depth[u]
            for v in self.children.get(u, []):
                if v not in depth:
                    depth[v] = du + 1
                    q.append(v)

        object.__setattr__(self, "parent_of", parent_of)
        object.__setattr__(self, "depth", depth)

    def primary_layer(self) -> List[str]:
        """Primary layer := children(root)."""
        return list(self.children.get(self.root, []))

    def descendant_leaves(self, node: str, leaf_ids: Set[str]) -> Set[str]:
        """
        Leaves under `node`, where "leaf" is defined by membership in `leaf_ids`.
        """
        if node not in self.depth:
            raise ValueError(f"Node '{node}' not found in hierarchy.")

        out: Set[str] = set()
        if node in leaf_ids:
            out.add(node)

        q = list(self.children.get(node, []))
        while q:
            x = q.pop()
            if x in leaf_ids:
                out.add(x)
            q.extend(self.children.get(x, []))
        return out

    def has_children(self, node: str) -> bool:
        return bool(self.children.get(node, []))

    def nodes(self) -> Set[str]:
        out: Set[str] = {self.root}
        for p, chs in self.children.items():
            out.add(p)
            out.update(chs)
        return out


# -----------------------
# SAMPLING UTILITIES
# -----------------------
def _softmax_sample_without_replacement(
    items: List[str],
    weights: Dict[str, float],
    rng: random.Random,
    temperature: float,
) -> List[str]:
    """
    Sequential sampling without replacement using a softmax over weights/temperature.
    Larger weight -> more likely to appear earlier.

    temperature:
      - 0.0  => deterministic (descending weight; random tie-break)
      - >0.0 => stochastic softmax sampling without replacement
    """
    if not items:
        return []
    if temperature <= 0.0:
        return sorted(items, key=lambda x: (-float(weights.get(x, 0.0)), rng.random()))

    remaining = items[:]
    perm: List[str] = []
    eps = 1e-12

    while remaining:
        logits = [float(weights.get(x, 0.0)) / max(temperature, eps) for x in remaining]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        z = sum(exps) + eps
        probs = [e / z for e in exps]

        u = rng.random()
        cdf = 0.0
        idx = len(remaining) - 1
        for j, p in enumerate(probs):
            cdf += p
            if u <= cdf:
                idx = j
                break

        perm.append(remaining.pop(idx))

    return perm


def _l1_normalize_abs(weights: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    """Return |w| / sum|w| (or uniform if all zero)."""
    keys = list(weights.keys())
    if not keys:
        return {}
    s = sum(abs(float(v)) for v in weights.values())
    if s <= eps:
        u = 1.0 / float(len(keys))
        return {k: u for k in keys}
    return {k: abs(float(weights[k])) / s for k in keys}


def _calibration_budget(K: int) -> int:
    """Default: 20% of K, at least 1."""
    if K <= 1:
        return 1
    return max(1, int(round(0.2 * K)))


def _validate_groups_nonempty(groups: Dict[str, List[str]]) -> None:
    if not groups:
        raise ValueError("groups is empty; cannot run Shapley.")
    for gid, leaves in groups.items():
        if leaves is None:
            raise ValueError(f"groups['{gid}'] is None; expected a list of leaf ids.")
        if len(leaves) == 0:
            raise ValueError(f"groups['{gid}'] is empty; each player must activate >=1 leaf.")


# -----------------------
# EPOCH SCHEDULING + WEIGHT UPDATES (INTERNAL ONLY)
# -----------------------
def _epoch_sizes_default(K: int) -> List[int]:
    """
    Split K permutations into multiple epochs.
    Default heuristic:
      - number of epochs ~ sqrt(K), capped, at least 2 when K>=2
      - epoch sizes differ by at most 1
    """
    if K <= 1:
        return [K]
    n_epochs = int(round(math.sqrt(K)))
    n_epochs = max(2, min(20, n_epochs))
    n_epochs = min(n_epochs, K)

    base = K // n_epochs
    rem = K % n_epochs
    sizes = [base + 1] * rem + [base] * (n_epochs - rem)
    return [s for s in sizes if s > 0]


def _blend_distributions(
    prev: Dict[str, float],
    new: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """
    out = (1-alpha)*prev + alpha*new, then L1-normalize(abs(.))
    alpha is clamped to [0,1].
    """
    if alpha < 0.0:
        alpha = 0.0
    elif alpha > 1.0:
        alpha = 1.0

    keys = set(prev.keys()) | set(new.keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = (1.0 - alpha) * float(prev.get(k, 0.0)) + alpha * float(new.get(k, 0.0))
    return _l1_normalize_abs(out)


def _mix_distributions(
    a: Dict[str, float],
    b: Dict[str, float],
    mix_b: float,
) -> Dict[str, float]:
    """
    out = (1-mix_b)*a + mix_b*b, then L1-normalize(abs(.))
    mix_b clamped to [0,1].
    """
    if mix_b < 0.0:
        mix_b = 0.0
    elif mix_b > 1.0:
        mix_b = 1.0

    keys = set(a.keys()) | set(b.keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = (1.0 - mix_b) * float(a.get(k, 0.0)) + mix_b * float(b.get(k, 0.0))
    return _l1_normalize_abs(out)


# -----------------------
# GROUP SHAPLEY (used for adaptive calibration)
# -----------------------
def _monte_carlo_group_shapley(
    *,
    score_fn: Callable[[Set[str]], float],
    groups: Dict[str, List[str]],
    K: int,
    rng: random.Random,
    verbose: bool = False,
    progress_cb: Optional[Callable[[int, int, float, float], Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    selection_temperature: float = 0.0,
) -> Dict[str, float]:
    """
    Monte Carlo Shapley–Shubik estimator over "group players".
    Adding player g activates all leaf features in groups[g].
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    _validate_groups_nonempty(groups)

    group_ids = list(groups.keys())
    phi = {gid: 0.0 for gid in group_ids}

    t0 = time.perf_counter()

    for k in range(K):
        if weights is not None and len(weights) > 0:
            perm = _softmax_sample_without_replacement(
                items=group_ids,
                weights=weights,
                rng=rng,
                temperature=selection_temperature,
            )
        else:
            perm = group_ids[:]
            rng.shuffle(perm)

        active: Set[str] = set()
        s_prev = score_fn(active)

        if verbose:
            mode = "biased" if (weights is not None and len(weights) > 0) else "uniform"
            print(f"  [HFR-TokenSHAP] {mode} GROUP perm {k+1:02d}/{K} start s={s_prev:+.6f}")

        for gid in perm:
            active.update(groups[gid])
            s_new = score_fn(active)
            phi[gid] += (s_new - s_prev)
            s_prev = s_new

        if progress_cb is not None:
            elapsed = time.perf_counter() - t0
            eta = _eta(elapsed, k + 1, K)
            progress_cb(k + 1, K, elapsed, eta)

    for gid in phi:
        phi[gid] /= float(K)

    return phi


# -----------------------
# MIXED MULTI-LEVEL "PLAYERS" VIA LEAF PERMUTATIONS
# -----------------------
def _sample_frontier_partition_nonadaptive(
    *,
    H: Hierarchy,
    leaf_set: Set[str],
    rng: random.Random,
    stop_probability: float,
    max_depth: Optional[int] = None,
) -> List[str]:
    """
    Sample a mixed-depth frontier (a partition of leaves into disjoint blocks),
    by recursively deciding to STOP at a node (use it as a block) or EXPAND into children.

    This returns a list of node-ids (blocks) whose descendant leaves form a partition of leaf_set.
    """
    if not (0.0 <= stop_probability <= 1.0):
        raise ValueError("stop_probability must be in [0,1].")

    desc_cache: Dict[str, Set[str]] = {}

    def desc(node: str) -> Set[str]:
        if node not in desc_cache:
            desc_cache[node] = H.descendant_leaves(node, leaf_set)
        return desc_cache[node]

    def rec(node: str) -> List[str]:
        leaves_here = desc(node)
        if not leaves_here:
            return []  # irrelevant branch

        # Atomic leaf => must stop.
        if node in leaf_set:
            return [node]

        chs = H.children.get(node, [])
        if not chs:
            return [node]  # fallback

        d = H.depth.get(node, 0)
        if max_depth is not None and d >= max_depth:
            return [node]

        # Decide stop vs expand
        if rng.random() < stop_probability:
            return [node]

        out: List[str] = []
        for c in chs:
            out.extend(rec(c))
        return out

    frontier: List[str] = []
    for p in H.primary_layer():
        frontier.extend(rec(p))

    # Safety: ensure cover (should hold by construction)
    covered: Set[str] = set()
    for b in frontier:
        covered.update(desc(b))
    missing = leaf_set - covered
    if missing:
        raise ValueError(f"Sampled frontier failed to cover leaves (bug). missing={sorted(missing)}")

    return frontier


def _sample_frontier_partition_adaptive(
    *,
    H: Hierarchy,
    leaf_set: Set[str],
    rng: random.Random,
    # initial weights on primary nodes (sum to 1)
    primary_weights: Dict[str, float],
    # how strongly weight controls expansion
    expand_base: float,
    expand_scale: float,
    max_depth: Optional[int] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Adaptive mixed-depth frontier sampling.

    We start with mass on primary nodes. Recursively:
      - If leaf => stop
      - Else expand with probability p_expand = clamp(expand_base + expand_scale * mass, 0..1)
      - If expand, mass is split uniformly across children that have leaves
      - If stop, the node becomes a block with that mass (used later to bias block order)

    Returns:
      frontier_blocks, block_masses
    """
    def clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    if expand_scale < 0.0:
        raise ValueError("expand_scale must be >= 0.")
    if not (0.0 <= expand_base <= 1.0):
        raise ValueError("expand_base must be in [0,1].")

    desc_cache: Dict[str, Set[str]] = {}

    def desc(node: str) -> Set[str]:
        if node not in desc_cache:
            desc_cache[node] = H.descendant_leaves(node, leaf_set)
        return desc_cache[node]

    block_masses: Dict[str, float] = {}

    def rec(node: str, mass: float) -> List[str]:
        leaves_here = desc(node)
        if not leaves_here:
            return []

        if node in leaf_set:
            block_masses[node] = block_masses.get(node, 0.0) + mass
            return [node]

        chs = H.children.get(node, [])
        if not chs:
            block_masses[node] = block_masses.get(node, 0.0) + mass
            return [node]

        d = H.depth.get(node, 0)
        if max_depth is not None and d >= max_depth:
            block_masses[node] = block_masses.get(node, 0.0) + mass
            return [node]

        p_expand = clamp01(expand_base + expand_scale * mass)
        if rng.random() >= p_expand:
            block_masses[node] = block_masses.get(node, 0.0) + mass
            return [node]

        # expand: split mass across effective children
        eff_children = [c for c in chs if desc(c)]
        if not eff_children:
            block_masses[node] = block_masses.get(node, 0.0) + mass
            return [node]

        share = mass / float(len(eff_children))
        out: List[str] = []
        for c in eff_children:
            out.extend(rec(c, share))
        return out

    frontier: List[str] = []
    # IMPORTANT: traverse *all* primary nodes to guarantee coverage even if some have zero mass
    for p in H.primary_layer():
        m = float(primary_weights.get(p, 0.0))
        frontier.extend(rec(p, m))

    # Ensure cover
    covered: Set[str] = set()
    for b in frontier:
        covered.update(desc(b))
    missing = leaf_set - covered
    if missing:
        raise ValueError(f"Adaptive sampled frontier failed to cover leaves (bug). missing={sorted(missing)}")

    # Normalize masses over blocks (for ordering bias later)
    block_masses = _l1_normalize_abs(block_masses)
    return frontier, block_masses


def _blocks_to_leaf_permutation(
    *,
    H: Hierarchy,
    frontier_blocks: List[str],
    leaf_set: Set[str],
    rng: random.Random,
    # ordering
    block_weights: Optional[Dict[str, float]] = None,
    selection_temperature: float = 0.0,
) -> List[str]:
    """
    Convert blocks (nodes) into a full LEAF permutation:
      1) order blocks (uniform or softmax-biased)
      2) within each block, shuffle its descendant leaves
      3) concatenate
    """
    # materialize block->leaves
    block_leaves: Dict[str, List[str]] = {}
    for b in frontier_blocks:
        ls = sorted(H.descendant_leaves(b, leaf_set))
        if not ls:
            continue
        block_leaves[b] = ls

    blocks = list(block_leaves.keys())
    if not blocks:
        raise ValueError("No non-empty blocks to permute (check leaf_set).")

    # order blocks
    if block_weights is not None and len(block_weights) > 0:
        ordered_blocks = _softmax_sample_without_replacement(
            items=blocks,
            weights=block_weights,
            rng=rng,
            temperature=selection_temperature,
        )
    else:
        ordered_blocks = blocks[:]
        rng.shuffle(ordered_blocks)

    # within-block shuffle -> leaves
    leaf_perm: List[str] = []
    for b in ordered_blocks:
        ls = block_leaves[b][:]
        rng.shuffle(ls)
        leaf_perm.extend(ls)

    # Safety: must be a permutation of leaf_set
    if set(leaf_perm) != set(leaf_set) or len(leaf_perm) != len(leaf_set):
        raise ValueError("Internal error: leaf permutation is not a bijection of leaf_set.")
    return leaf_perm


def _monte_carlo_leaf_shapley_from_leaf_perms(
    *,
    score_fn: Callable[[Set[str]], float],
    leaf_ids: List[str],
    K: int,
    rng: random.Random,
    leaf_perm_sampler: Callable[[], List[str]],
    verbose: bool = False,
    progress_cb: Optional[Callable[[int, int, float, float], Any]] = None,
) -> Dict[str, float]:
    """
    Leaf-level Shapley estimator via Monte Carlo over LEAF permutations.

    The sampler can restrict permutations (e.g., via hierarchical blocks).
    This returns attribution for every leaf in leaf_ids.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    leaf_set = set(leaf_ids)
    if not leaf_set:
        raise ValueError("leaf_ids is empty.")

    phi = {lf: 0.0 for lf in leaf_ids}
    t0 = time.perf_counter()

    for k in range(K):
        perm = leaf_perm_sampler()
        if verbose:
            print(f"  [HFR-TokenSHAP] LEAF perm {k+1:02d}/{K} (len={len(perm)})")
            print(f"[DEBUG] Shapley iter {k+1:02d}: leaf permutation = {perm}")

        active: Set[str] = set()
        s_prev = score_fn(active)

        for lf in perm:
            active.add(lf)
            s_new = score_fn(active)
            phi[lf] += (s_new - s_prev)
            s_prev = s_new

        if progress_cb is not None:
            elapsed = time.perf_counter() - t0
            eta = _eta(elapsed, k + 1, K)
            progress_cb(k + 1, K, elapsed, eta)

    for lf in phi:
        phi[lf] /= float(K)
    return phi


# -----------------------
# Monte Carlo HFR-TokenSHAP
# -----------------------
def monte_carlo_hfr_tokenshap(
    score_fn: Callable[[Set[str]], float],
    feature_ids: List[str],
    hierarchy_children: Optional[Dict[str, List[str]]] = None,
    root: Optional[str] = None,
    leaf_ids: Optional[List[str]] = None,
    # --- fixed-cut / custom groups (legacy modes) ---
    cut_nodes: Optional[List[str]] = None,
    groups: Optional[Dict[str, List[str]]] = None,
    # --- MC ---
    K: int = 10,
    seed: int = 0,
    verbose: bool = False,
    progress_cb: Optional[Callable[[int, int, float, float], Any]] = None,
    # --- Adaptivity (legacy group-bias OR mixed mode) ---
    adaptive_search: bool = True,
    selection_temperature: float = 0.0,
    adaptive_calibration_K: Optional[int] = None,
    # --- Mixed multi-level player mixing ---
    mixed_players: bool = True,
    # non-adaptive mixing control (probability to STOP at an internal node => make a block)
    mix_stop_probability: float = 0.5,
    mix_max_depth: Optional[int] = None,
    # adaptive mixing control (how aggressively important mass triggers expansion)
    adaptive_expand_base: float = 0.2,
    adaptive_expand_scale: float = 1.6,
    # for debugging / analysis
    return_level_phis: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Dict[str, float]]]]:
    """
    HFR-TokenSHAP estimator.

    IMPORTANT: With mixed_players=True, we return LEAF-level attributions (phi per leaf),
    computed from Monte Carlo over LEAF permutations generated by hierarchical block mixing.

    Modes:

    1) mixed_players=False (legacy behavior)
       - non-adaptive: fixed frontier via cut_nodes or groups, else flat

    2) mixed_players=True
       - non-adaptive: each permutation samples a random mixed-depth frontier (partition into blocks),
         then generates a leaf permutation by ordering blocks uniformly and shuffling within-block.
       - adaptive:
           * initial calibration on primary layer
           * then: SEQUENTIAL EPOCHS (frozen sampling within epoch)
             where primary mass is updated cumulatively using:
               - attribution evidence (from accumulated leaf φ)
               - sampling-mass evidence (from accumulated block masses)
             between epochs.

    Return:
      - if return_level_phis=False: returns phi (leaf-level if mixed_players=True; else frontier-level legacy)
      - if return_level_phis=True : returns (phi, level_phis) where level_phis are calibration phis (adaptive only)
    """
    if K < 1:
        raise ValueError("K must be >= 1")

    rng = random.Random(seed)

    leaf_universe: List[str] = leaf_ids if leaf_ids is not None else feature_ids
    if not leaf_universe:
        raise ValueError("leaf_ids/feature_ids is empty; cannot run.")

    # Preserve a deterministic leaf ordering (avoid Python set iteration nondeterminism).
    seen: Set[str] = set()
    leaf_order: List[str] = []
    for x in leaf_universe:
        if x not in seen:
            seen.add(x)
            leaf_order.append(x)
    leaf_set: Set[str] = seen
    if not leaf_set:
        raise ValueError("leaf_ids/feature_ids is empty; cannot run.")

    # ------------------------------------------------------------------
    # MIXED MULTI-LEVEL PLAYER MIXING (LEAF PERMUTATION BASED)
    # ------------------------------------------------------------------
    if mixed_players:
        if hierarchy_children is None:
            # Flat leaf permutations (TokenSHAP-style)
            def sampler_flat() -> List[str]:
                perm = leaf_order[:]
                rng.shuffle(perm)
                return perm

            phi = _monte_carlo_leaf_shapley_from_leaf_perms(
                score_fn=score_fn,
                leaf_ids=leaf_order,
                K=K,
                rng=rng,
                leaf_perm_sampler=sampler_flat,
                verbose=verbose,
                progress_cb=progress_cb,
            )
            return phi if not return_level_phis else (phi, [])

        if root is None:
            root = infer_root(hierarchy_children)
        H = Hierarchy(children=hierarchy_children, root=root)

        # Guard: if the hierarchy root is also listed as a leaf feature, the "leaves" set is not a
        # standard tree-leaf set. The mixed-depth partitioning relies on leaves being below the root.
        if H.root in leaf_set:
            if len(leaf_set) == 1:
                # Single-feature case is well-defined.
                s0 = score_fn(set())
                s1 = score_fn({H.root})
                phi_single = {H.root: (s1 - s0)}
                return phi_single if not return_level_phis else (phi_single, [])
            raise ValueError(
                "Invalid leaf_ids: includes the hierarchy root alongside other leaves. "
                "This implementation assumes leaves are strictly below the root."
            )

        level_phis: List[Dict[str, float]] = []

        # ---- Adaptive mixed mode (NOW epoch-based and cumulative) ----
        if adaptive_search:
            primary = H.primary_layer()
            if not primary:
                raise ValueError("Primary layer is empty; cannot run adaptive.")

            # groups for primary nodes
            groups_primary: Dict[str, List[str]] = {}
            primary_to_leaves: Dict[str, List[str]] = {}
            for p in primary:
                ls = sorted(H.descendant_leaves(p, leaf_set))
                if ls:
                    groups_primary[p] = ls
                    primary_to_leaves[p] = ls
            _validate_groups_nonempty(groups_primary)

            K_calib = adaptive_calibration_K if adaptive_calibration_K is not None else _calibration_budget(K)

            if verbose:
                print(
                    f"[Adaptive-MIXED] initial calib primary |G|={len(groups_primary)} "
                    f"K_calib={K_calib} K_final={K}"
                )

            # Initial calibration (kept as before)
            phi_primary = _monte_carlo_group_shapley(
                score_fn=score_fn,
                groups=groups_primary,
                K=K_calib,
                rng=rng,
                verbose=verbose,
                progress_cb=None,
                weights=None,
                selection_temperature=0.0,
            )
            level_phis.append(phi_primary)

            # Start primary weights from calibration
            primary_weights: Dict[str, float] = _l1_normalize_abs(phi_primary)

            # ---- Epoch-based adaptive resampling (frozen within epoch) ----
            epoch_sizes = _epoch_sizes_default(K)

            # Evidence accumulators (CUMULATIVE across all epochs/perms)
            phi_sum: Dict[str, float] = {lf: 0.0 for lf in leaf_order}  # unnormalized sum
            primary_mass_sum: Dict[str, float] = {p: 0.0 for p in groups_primary.keys()}

            # Cache: map any node -> its primary ancestor (first child of root on its path)
            primary_ancestor_cache: Dict[str, str] = {}

            def primary_of(node: str) -> str:
                if node in primary_ancestor_cache:
                    return primary_ancestor_cache[node]
                # climb until parent is root; if node is a primary itself, return it
                cur = node
                if cur == H.root:
                    raise ValueError("Unexpected: frontier block is root (should be below root).")
                while True:
                    par = H.parent_of.get(cur, None)
                    if par is None:
                        # cur might be a primary (no parent entry) only if it is root; already handled.
                        raise ValueError(f"Node '{node}' has no parent; hierarchy inconsistent.")
                    if par == H.root:
                        primary_ancestor_cache[node] = cur
                        return cur
                    cur = par

            # How much to include sampling-mass evidence vs attribution evidence in weight updates.
            # (Small value -> mostly attribution-driven; still explicitly uses cumulative masses.)
            MASS_MIX_DEFAULT = 0.25

            t0 = time.perf_counter()
            done = 0  # permutations completed (out of K)

            for e, K_epoch in enumerate(epoch_sizes, start=1):
                # Freeze weights for this epoch
                epoch_primary_weights = dict(primary_weights)

                if verbose:
                    print(
                        f"[Adaptive-MIXED] epoch {e:02d}/{len(epoch_sizes)} "
                        f"K_epoch={K_epoch} (done={done}/{K})"
                    )

                for _ in range(K_epoch):
                    # Sample a frontier + block masses using the *epoch-frozen* weights
                    frontier_blocks, block_masses = _sample_frontier_partition_adaptive(
                        H=H,
                        leaf_set=leaf_set,
                        rng=rng,
                        primary_weights=epoch_primary_weights,
                        expand_base=adaptive_expand_base,
                        expand_scale=adaptive_expand_scale,
                        max_depth=mix_max_depth,
                    )

                    # Accumulate sampling-mass evidence (cumulative, all epochs)
                    for b, w in block_masses.items():
                        p = primary_of(b)
                        if p in primary_mass_sum:
                            primary_mass_sum[p] += abs(float(w))

                    # Convert to a leaf permutation (ordering biased by per-sample block masses)
                    perm = _blocks_to_leaf_permutation(
                        H=H,
                        frontier_blocks=frontier_blocks,
                        leaf_set=leaf_set,
                        rng=rng,
                        block_weights=block_masses,
                        selection_temperature=selection_temperature,
                    )

                    if verbose:
                        print(f"  [HFR-TokenSHAP] LEAF perm {done+1:02d}/{K} (len={len(perm)})")
                        print(f"[DEBUG] Shapley iter {done+1:02d}: leaf permutation = {perm}")

                    # Shapley accumulation for this permutation
                    active: Set[str] = set()
                    s_prev = score_fn(active)

                    for lf in perm:
                        active.add(lf)
                        s_new = score_fn(active)
                        phi_sum[lf] += (s_new - s_prev)
                        s_prev = s_new

                    done += 1

                    if progress_cb is not None:
                        elapsed = time.perf_counter() - t0
                        eta = _eta(elapsed, done, K)
                        progress_cb(done, K, elapsed, eta)

                # Between epochs: re-estimate masses using ALL evidence so far
                if done < K:
                    # Attribution evidence: aggregate |phi_sum| within each primary subtree
                    attr_scores: Dict[str, float] = {}
                    for p, leaves_p in primary_to_leaves.items():
                        s = 0.0
                        for lf in leaves_p:
                            s += abs(float(phi_sum.get(lf, 0.0)))
                        attr_scores[p] = s
                    attr_norm = _l1_normalize_abs(attr_scores)

                    # Sampling-mass evidence (cumulative)
                    mass_norm = _l1_normalize_abs(primary_mass_sum)

                    # Combine attribution + mass evidence, then smooth update
                    combined = _mix_distributions(attr_norm, mass_norm, mix_b=MASS_MIX_DEFAULT)

                    # Smoothing schedule: early updates more responsive, later more stable
                    # (epoch index e starts at 1)
                    alpha = min(0.85, 1.0 / math.sqrt(float(e)))

                    primary_weights = _blend_distributions(primary_weights, combined, alpha=alpha)

            # Final average over K permutations (leaf-level)
            phi = {lf: (phi_sum[lf] / float(K)) for lf in leaf_order}
            return phi if not return_level_phis else (phi, level_phis)

        # ---- Non-adaptive mixed mode ----
        def sampler_nonadaptive() -> List[str]:
            frontier_blocks = _sample_frontier_partition_nonadaptive(
                H=H,
                leaf_set=leaf_set,
                rng=rng,
                stop_probability=mix_stop_probability,
                max_depth=mix_max_depth,
            )
            return _blocks_to_leaf_permutation(
                H=H,
                frontier_blocks=frontier_blocks,
                leaf_set=leaf_set,
                rng=rng,
                block_weights=None,
                selection_temperature=0.0,
            )

        phi = _monte_carlo_leaf_shapley_from_leaf_perms(
            score_fn=score_fn,
            leaf_ids=leaf_order,
            K=K,
            rng=rng,
            leaf_perm_sampler=sampler_nonadaptive,
            verbose=verbose,
            progress_cb=progress_cb,
        )
        return phi if not return_level_phis else (phi, [])

    # ------------------------------------------------------------------
    # LEGACY BEHAVIOR
    # ------------------------------------------------------------------
    if adaptive_search:
        raise ValueError(
            "adaptive_search=True with mixed_players=False is not implemented in this revised script. "
            "Use mixed_players=True for multi-level mixing, or reinsert your legacy adaptive logic."
        )

    # Non-adaptive legacy: groups / fixed cut / flat groups
    if groups is not None:
        _validate_groups_nonempty(groups)
        return _monte_carlo_group_shapley(
            score_fn=score_fn,
            groups=groups,
            K=K,
            rng=rng,
            verbose=verbose,
            progress_cb=progress_cb,
            weights=None,
            selection_temperature=0.0,
        )

    if hierarchy_children is not None or cut_nodes is not None:
        if hierarchy_children is None or cut_nodes is None:
            raise ValueError(
                "Fixed-cut HFR mode requires BOTH hierarchy_children and cut_nodes "
                "(or provide groups explicitly)."
            )
        if root is None:
            root = infer_root(hierarchy_children)
        H = Hierarchy(children=hierarchy_children, root=root)

        # Convert fixed cut into groups (must cover all leaves & no overlaps)
        fixed_groups: Dict[str, List[str]] = {}
        assigned: Dict[str, str] = {}

        for g in cut_nodes:
            leaves_g = H.descendant_leaves(g, leaf_set)
            if not leaves_g:
                continue
            for lf in leaves_g:
                if lf in assigned:
                    raise ValueError(
                        f"Invalid cut: leaf '{lf}' covered by multiple nodes "
                        f"('{assigned[lf]}' and '{g}')."
                    )
                assigned[lf] = g
            fixed_groups[g] = sorted(leaves_g)

        missing = leaf_set - set(assigned.keys())
        if missing:
            raise ValueError(f"Cut does not cover all leaves: missing={sorted(missing)}")

        return _monte_carlo_group_shapley(
            score_fn=score_fn,
            groups=fixed_groups,
            K=K,
            rng=rng,
            verbose=verbose,
            progress_cb=progress_cb,
            weights=None,
            selection_temperature=0.0,
        )

    # fallback: flat TokenSHAP style (each feature is its own group)
    flat_groups = {fid: [fid] for fid in leaf_order}
    return _monte_carlo_group_shapley(
        score_fn=score_fn,
        groups=flat_groups,
        K=K,
        rng=rng,
        verbose=verbose,
        progress_cb=progress_cb,
        weights=None,
        selection_temperature=0.0,
    )


# -----------------------
# REPEATS
# -----------------------
def shapley_with_repeats(
    score_fn: Callable[[Set[str]], float],
    feature_ids: List[str],
    hierarchy_children: Optional[Dict[str, List[str]]] = None,
    root: Optional[str] = None,
    leaf_ids: Optional[List[str]] = None,
    cut_nodes: Optional[List[str]] = None,
    groups: Optional[Dict[str, List[str]]] = None,
    K: int = 10,
    runs: int = 3,
    seed: int = 0,
    verbose: bool = False,
    progress_cb: Optional[Callable[[int, int, int, int, float, float], Any]] = None,
    adaptive_search: bool = True,
    selection_temperature: float = 0.0,
    adaptive_calibration_K: Optional[int] = None,
    mixed_players: bool = True,
    mix_stop_probability: float = 0.5,
    mix_max_depth: Optional[int] = None,
    adaptive_expand_base: float = 0.2,
    adaptive_expand_scale: float = 1.6,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Repeat estimator multiple times and return (mean_phi, std_phi).

    If mixed_players=True => phi keys are leaf ids.
    Else => legacy keys (groups/cut nodes/feature ids).
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")

    all_phis: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    for r in range(runs):

        def perm_progress(perm_done: int, K_eff: int, elapsed_perm: float, eta_perm: float):
            if progress_cb is None:
                return
            elapsed_total = time.perf_counter() - t0
            total_work = runs * K
            done_work = r * K + perm_done
            eta_total = _eta(elapsed_total, done_work, total_work)
            progress_cb(r + 1, runs, perm_done, K_eff, elapsed_total, eta_total)

        phi_r = monte_carlo_hfr_tokenshap(
            score_fn=score_fn,
            feature_ids=feature_ids,
            hierarchy_children=hierarchy_children,
            root=root,
            leaf_ids=leaf_ids,
            cut_nodes=cut_nodes,
            groups=groups,
            K=K,
            seed=seed + 9973 * r,
            verbose=verbose,
            progress_cb=perm_progress,
            adaptive_search=adaptive_search,
            selection_temperature=selection_temperature,
            adaptive_calibration_K=adaptive_calibration_K,
            mixed_players=mixed_players,
            mix_stop_probability=mix_stop_probability,
            mix_max_depth=mix_max_depth,
            adaptive_expand_base=adaptive_expand_base,
            adaptive_expand_scale=adaptive_expand_scale,
            return_level_phis=False,
        )
        all_phis.append(phi_r)

    keys = list(all_phis[0].keys())
    mean_phi: Dict[str, float] = {k: 0.0 for k in keys}
    std_phi: Dict[str, float] = {k: 0.0 for k in keys}

    for k in keys:
        vals = [d[k] for d in all_phis]
        m = sum(vals) / len(vals)
        mean_phi[k] = m
        if len(vals) == 1:
            std_phi[k] = 0.0
        else:
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            std_phi[k] = math.sqrt(var)

    return mean_phi, std_phi
