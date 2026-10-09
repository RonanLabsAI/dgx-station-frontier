# Derived from original-el8/dgx-station-gb300-research @ 7699ae54, deepseek-v4.1-flash/m3/hook/mega_peer_hook.py
# Copyright 2026 Jason Cook. Licensed under the Apache License 2.0.
# Modifications Copyright 2026 RonanLabs (Apache License 2.0): rewritten for DeepSeek-V4-Pro-0813 in vLLM
# nightly-af7f9488, where deepseek_v4 without MegaMoE builds its experts with the generic FusedMoE factory
# (layers.N.ffn.experts.routed_experts, a RoutedExperts module) and the marlin MXFP4 kernel.
"""E3 warm tier for DeepSeek-V4-Pro on TP2+EP2 GB300 Stations: RTX PRO 6000 sidecar per EP rank.

Three tiers per Station (per EP rank, local experts only):
  hot   HBM    - whatever the GB300 already holds in HBM (E1: positional, layers 34-60; E2: hot rows per layer)
  warm  6000   - the rowmap's "warm" experts of each hooked layer, computed by the sidecar on the RTX PRO 6000
  cold  Grace  - the rest of the UVA-offloaded experts, still read over C2C by the GB300's marlin kernel

How the warm tier is taken off the GB300 without touching weight placement: after loading, each hooked layer's
expert map (global id -> local slot, -1 = not mine) is copied and the warm experts are set to -1. vLLM's marlin path
(moe_align_block_size(ignore_invalid_experts=True) + ops.moe_sum(expert_map)) then skips those routes, so their
weights are never read from Grace. The warm routes go to the sidecar instead and come back as one add per token.
The GB300 keeps every weight, so:
  - E3_ENABLE=0, or a batch over E3_MAX_ROWS tokens, falls back to the full map = stock E1 numerics;
  - the cosine gate (E3_CHECK=N) compares the sidecar's rows against marlin on the same warm experts, using a
    warm-only map, on eager calls with T >= E3_CHECK_MIN_T. N is a per-layer budget, and a check with an all-zero
    reference (vLLM's dummy profile run) does not consume it.

Each EP rank's sidecar serves only that rank's local experts (linear placement: rank r owns [192r, 192r+192)).
The rowmap lists global ids; each side keeps the ids it owns, sorted, and the sorted position is the wire id.
A sha256 of that per-layer list is the handshake: the worker refuses a sidecar built from a different rowmap.

Composes with E2: if an E2 hook splits a layer's local experts into an HBM module and a Grace module, apply this
hook to the layer's RoutedExperts module(s) the same way; the warm experts must then be absent from both.

Env:
  E3_HOOK=/path/e3_hook.py (sitecustomize installs it)   E3_ROWMAP=rowmap.json   E3_ENABLE=1
  E3_CHECK=0 (per-layer check budget)   E3_CHECK_MIN_T=256   E3_COUNT=/prof/e3-counts.json (route histogram)
  E3_MAX_ROWS=8192   E3_SMALL_ROWS=64   E3_PEER_SPINS=20000000   E3_SHM=/dev/shm/e3_dsv4pro_peer
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import re
import sys
import threading
import time

import torch

E = 384
TOPK = 6
HIDDEN = 7168
ENABLE = os.environ.get("E3_ENABLE", "1") == "1"
ROWMAP_PATH = os.environ.get("E3_ROWMAP", "/e3/rowmap.json")
CHECK = int(os.environ.get("E3_CHECK", "0"))
CHECK_MIN_T = int(os.environ.get("E3_CHECK_MIN_T", "256"))
# vLLM's dummy profile run uses exactly max_num_batched_tokens (8192) tokens; keep it out of the gate.
CHECK_MAX_T = int(os.environ.get("E3_CHECK_MAX_T", "8191"))
COUNT_PATH = os.environ.get("E3_COUNT", "")
STATS_PATH = os.environ.get("E3_STATS", "")
TARGET = "vllm.model_executor.layers.fused_moe.routed_experts"
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts(?:\.routed_experts)?$")

_log = lambda m: sys.stderr.write(f"E3 {m}\n")  # noqa: E731

_rowmap = None
_peer = None
_pt = None
_orig = {}
_states: dict[int, "LayerState"] = {}
_count_host = None
_count_copied = 0.0
_check_log: list = []


def _load_rowmap() -> dict:
    global _rowmap
    if _rowmap is None:
        with open(ROWMAP_PATH) as f:
            _rowmap = json.load(f)
        _log(f"rowmap {ROWMAP_PATH}: {len(_rowmap['layers'])} layers, "
             f"{sum(len(v.get('warm', [])) for v in _rowmap['layers'].values())} warm experts (both ranks)")
    return _rowmap


def _peer_mod():
    global _pt
    if _pt is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import peer_tier_v4

        _pt = peer_tier_v4
    return _pt


class LayerState:
    def __init__(self, idx: int):
        self.idx = idx
        self.ready = False
        self.warm: list[int] = []        # global ids, sorted, local to this rank
        self.lut = None                  # int32 [E]: global id -> wire id (sorted position) or -1
        self.map_full = None             # the layer's original expert map
        self.map_prod = None             # warm experts masked to -1
        self.map_warm = None             # only the warm experts (cosine reference)
        self.counts = None               # int64 [E] routed tokens per global expert (E3_COUNT)
        self.checks_left = CHECK
        self.offloaded = False
        self.e2 = None                   # E2 PinState of this layer, when E2's split hook is installed
        self.e2_cold_full = self.e2_cold_prod = self.e2_cold_warm = self.e2_hot_none = None


def _layer_idx(name: str):
    if name.startswith("mtp") or ".mtp." in name:
        return None
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


def _init_peer(device):
    global _peer
    if _peer is not None:
        return _peer
    pt = _peer_mod()
    peer = pt.PeerTierV4(device)
    for _ in range(600):  # the launcher starts the sidecar first; allow it up to 10 min to finish preparing
        ok, h, rank = peer.serving()
        if ok:
            break
        time.sleep(1)
    else:
        raise RuntimeError("E3: sidecar never reported serving on " + pt.PATH)
    _peer = peer
    _log(f"peer on {pt.PATH}: sidecar serving ep_rank={rank} hash={h:#x} max_rows={peer.max_rows} "
         f"timeout_spins={peer.timeout}")
    if STATS_PATH:
        _start_stats_dumper()
    return peer


def _setup(layer, st: LayerState, device) -> None:
    pt = _peer_mod()
    em = layer._expert_map
    if em is None:
        raise RuntimeError("E3 needs expert parallel (no expert map on " + layer.layer_name + ")")
    em_cpu = em.detach().to("cpu", torch.int64)
    local = (em_cpu >= 0).nonzero().flatten().tolist()
    n_local = len(local)
    ep_rank = local[0] // n_local
    if local != list(range(ep_rank * n_local, (ep_rank + 1) * n_local)):
        raise RuntimeError(f"E3: layer {st.idx} expert map is not linear placement")
    rowmap = _load_rowmap()
    warm_all = pt.rank_warm(rowmap, ep_rank, n_local)
    peer = _init_peer(device)
    ok, h, srank = peer.serving()
    if srank != ep_rank or h != pt.warm_hash(warm_all):
        raise RuntimeError(f"E3: sidecar serves rank {srank} hash {h:#x}; this worker is rank {ep_rank} "
                           f"hash {pt.warm_hash(warm_all):#x} (rowmap {ROWMAP_PATH})")
    st.warm = warm_all.get(st.idx, [])
    if peer.warm_count(st.idx) != len(st.warm):
        raise RuntimeError(f"E3: layer {st.idx}: sidecar has {peer.warm_count(st.idx)} warm, worker {len(st.warm)}")
    lut = torch.full((E,), -1, dtype=torch.int32)
    for i, g in enumerate(st.warm):
        lut[g] = i
    st.lut = lut.to(device)
    st.map_full = em
    prod = em.clone()
    warm_only = torch.full_like(em, -1)
    if st.warm:
        w = torch.tensor(st.warm, dtype=torch.long, device=em.device)
        prod[w] = -1
        warm_only[w] = em[w]
    st.map_prod, st.map_warm = prod, warm_only
    e2 = sys.modules.get("dsv4pro_pin_hook")
    if e2 is not None and getattr(e2, "MODE", "") == "split":
        st2 = e2._STATES.get(layer.w13_weight.data_ptr())
        if st2 is None:
            raise RuntimeError(f"E3: E2 split hook is on but layer {st.idx} has no E2 state")
        st.e2 = st2
        st.e2_cold_full = st2.map_cold
        st.e2_hot_none = torch.full_like(st2.map_hot, -1)
        st.e2_cold_prod = st2.map_cold.clone()
        st.e2_cold_warm = torch.full_like(st2.map_cold, -1)
        if st.warm:
            w = torch.tensor(st.warm, dtype=torch.long, device=st2.map_cold.device)
            if bool((st2.map_hot[w] >= 0).any()) or bool((st2.map_cold[w] < 0).any()):
                raise RuntimeError(f"E3: layer {st.idx}: warm experts must be E2-cold (rowmap built without "
                                   "--e2-rowmap?)")
            st.e2_cold_prod[w] = -1
            st.e2_cold_warm[w] = st2.map_cold[w]
            st2.map_cold = st.e2_cold_prod
    if st.warm:
        layer._expert_map = st.map_prod
    st.offloaded = bool(getattr(getattr(layer, "w13_weight", None), "_vllm_is_uva_offloaded", False))
    if COUNT_PATH:
        st.counts = torch.zeros(E, dtype=torch.int64, device=device)
    st.ready = True
    _states[st.idx] = st
    if st.warm and not st.offloaded:
        _log(f"WARNING layer {st.idx}: {len(st.warm)} warm experts but w13 is NOT UVA-offloaded (HBM layer)")
    if st.idx in (0, 33, 60) or (st.warm and st.idx == min(warm_all)):
        _log(f"layer {st.idx} ({layer.layer_name}): ep_rank {ep_rank}, {n_local} local, {len(st.warm)} warm on "
             f"the 6000, w13 offloaded={st.offloaded}, quant={type(layer.quant_method).__name__}, "
             f"e2={'split' if st.e2 is not None else 'off'}")


def _start_count_dumper():
    """Write the pinned host copy of the counts every 20 s. The thread makes no CUDA calls (a CUDA call from another
    thread during vLLM's graph capture invalidates the capture)."""
    global _count_host
    _count_host = torch.zeros(64, E, dtype=torch.int64, device="cpu", pin_memory=True)

    def loop():
        while True:
            time.sleep(20)
            try:
                data = {str(i): _count_host[i].tolist() for i, s in sorted(_states.items()) if s.counts is not None}
                tmp = COUNT_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"format": "routing-counts/v1", "source": "e3_hook E3_COUNT", "time": time.time(),
                               "layers": data}, f)
                os.replace(tmp, COUNT_PATH)
            except Exception as exc:  # keep serving
                _log(f"count dump failed: {exc!r}")

    threading.Thread(target=loop, daemon=True, name="e3-count-dump").start()
    _log(f"route counting on, dumping to {COUNT_PATH}")


def _start_stats_dumper():
    def loop():
        while True:
            time.sleep(10)
            try:
                tmp = STATS_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"time": time.time(), "peer": _peer.counters(), "checks": _check_log[-512:]}, f)
                os.replace(tmp, STATS_PATH)
            except Exception as exc:
                _log(f"stats dump failed: {exc!r}")

    threading.Thread(target=loop, daemon=True, name="e3-stats-dump").start()


def _maybe_copy_counts() -> None:
    """Engine thread, eager calls only, at most every 10 s: async copy of every layer's counts to pinned host."""
    global _count_copied
    if _count_host is None or torch.cuda.is_current_stream_capturing():
        return
    now = time.time()
    if now - _count_copied < 10:
        return
    _count_copied = now
    for i, s in _states.items():
        if s.counts is not None:
            _count_host[i].copy_(s.counts, non_blocking=True)


def _call(layer, emap, x, topk_weights, topk_ids, shared_experts, shared_experts_input, e2_maps=None):
    """Run the stock path with a different expert map (and, under E2, different hot/cold maps). Eager only."""
    st = layer._e3
    saved = layer._expert_map
    layer._expert_map = emap
    e2_saved = None
    if st and st.e2 is not None and e2_maps is not None:
        e2_saved = (st.e2.map_hot, st.e2.map_cold)
        st.e2.map_hot, st.e2.map_cold = e2_maps
    try:
        return _orig["forward_modular"](layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    finally:
        layer._expert_map = saved
        if e2_saved is not None:
            st.e2.map_hot, st.e2.map_cold = e2_saved


def _forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
    st = getattr(self, "_e3", None)
    if st is None:
        idx = _layer_idx(getattr(self, "layer_name", ""))
        st = LayerState(idx) if idx is not None else False
        self._e3 = st
    if st is False or not ENABLE:
        return _orig["forward_modular"](self, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    capturing = torch.cuda.is_current_stream_capturing()
    if not st.ready:
        if capturing:
            raise RuntimeError(f"E3: layer {st.idx} first seen during CUDA graph capture; cannot set up there")
        if st.idx in {int(k) for k in _load_rowmap()["layers"]} or COUNT_PATH:
            _setup(self, st, x.device)
            if COUNT_PATH and _count_host is None:
                _start_count_dumper()
        else:
            st.ready = True
    T = x.shape[0]
    if st.counts is not None:
        valid = topk_ids >= 0
        st.counts.index_add_(0, topk_ids.clamp_min(0).flatten().long(), valid.flatten().long())
        _maybe_copy_counts()
    if not st.warm or T == 0:
        return _orig["forward_modular"](self, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    if self._expert_map is not st.map_prod:
        # Something re-registered the expert map (EPLB, a config reset): re-mask before it double-counts.
        if capturing:
            raise RuntimeError(f"E3: layer {st.idx} expert map replaced; refusing to capture")
        _log(f"layer {st.idx}: expert map was replaced; re-masking warm experts")
        st.map_full = self._expert_map
        st.map_prod = st.map_full.clone()
        st.map_prod[torch.tensor(st.warm, device=st.map_full.device)] = -1
        self._expert_map = st.map_prod
    peer = _peer
    if T > peer.max_rows:
        # Too many rows for the shared buffer: the GB300 still holds the warm weights, so run stock E1 numerics.
        e2m = (st.e2.map_hot, st.e2_cold_full) if st.e2 is not None else None
        return _call(self, st.map_full, x, topk_weights, topk_ids, shared_experts, shared_experts_input, e2m)

    xs = x[:, :HIDDEN] if x.shape[-1] != HIDDEN else x
    ids = topk_ids.clamp_min(0).long()
    wid = torch.where(topk_ids >= 0, st.lut[ids], -1)
    has, pos, rows = peer.send(xs, wid, topk_weights, st.idx)
    y = _orig["forward_modular"](self, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"E3: layer {st.idx}: routed experts returned {type(y)} (deferred finalize?); "
                           "the add needs a finalized tensor")
    check = st.checks_left > 0 and CHECK_MIN_T <= T <= CHECK_MAX_T and not capturing
    ys = y[:, :HIDDEN] if y.shape[-1] != HIDDEN else y
    peer.finish(ys, has, pos, rows)
    if check:
        _cosine_check(self, st, x, topk_weights, topk_ids, shared_experts, shared_experts_input, has, pos, rows, y)
    return y


def _cosine_check(layer, st, x, topk_weights, topk_ids, se, sei, has, pos, rows, y):
    peer = _peer
    torch.cuda.synchronize()
    T = x.shape[0]
    e2m = (st.e2_hot_none, st.e2_cold_warm) if st.e2 is not None else None
    ref = _call(layer, st.map_warm, x, topk_weights, topk_ids, se, sei, e2m)
    ref = (ref[:, :HIDDEN] if ref.shape[-1] != HIDDEN else ref).float()
    ref_norm = float(ref.norm())
    n_rows = int(rows.item())
    if ref_norm < 1e-6 or n_rows == 0:
        return  # dummy profile input or no warm route: does not consume the budget
    ok = int(peer.ok.item())
    side = peer.dense_rows(T, has, pos) if ok else torch.zeros_like(ref)
    diff = (side - ref).flatten()
    cos = float(torch.nn.functional.cosine_similarity(side.flatten(), ref.flatten(), dim=0))
    rel = float(diff.norm() / max(ref_norm, 1e-9))
    share = ref_norm / max(float(y.float().norm()), 1e-9)
    st.checks_left -= 1
    rec = {"layer": st.idx, "T": T, "rows": n_rows, "ok": ok, "cos": round(cos, 6), "rel": round(rel, 5),
           "warm_share": round(share, 4)}
    _check_log.append(rec)
    _log(f"check layer={st.idx} T={T} rows={n_rows} ok={ok} cos={cos:.6f} rel_diff={rel:.5f} "
         f"(warm share of routed output norm {share:.3f})")


def _patch(module):
    cls = module.RoutedExperts
    if getattr(cls, "_e3_patched", False):
        return
    _orig["forward_modular"] = cls.forward_modular
    cls.forward_modular = _forward_modular
    cls._e3_patched = True
    _log(f"patched {TARGET}.RoutedExperts.forward_modular (enable={ENABLE}, check={CHECK}/layer, "
         f"min_t={CHECK_MIN_T}, count={'on' if COUNT_PATH else 'off'})")


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch):
        self.target, self.patch = target, patch

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig_exec=orig_exec, _patch_fn=self.patch):
            _orig_exec(module)
            _patch_fn(module)

        spec.loader.exec_module = exec_module
        return spec


def install():
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    else:
        sys.meta_path.insert(0, _Finder(TARGET, _patch))
