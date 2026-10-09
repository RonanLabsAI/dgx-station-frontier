# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# Portions ported from J-M-Recipes/recipes pin_hot_experts_hook v15 @71ea3eef, Copyright 2026 J&M Recipes,
# MIT License (see licenses/J-M-Recipes-MIT.txt). The MIT notice is retained as required.
"""dsv4pro_pin_hook.py -- E2: pin-hot-experts for DeepSeek-V4-Pro-0813 on vLLM af7f9488, MARLIN MXFP4 MoE, TP2+EP2.

A port of the J-M-Recipes pin_hot_experts_hook (v15, MIT, J-M-Recipes/recipes @ 71ea3eef) to:
  * the Marlin MoE path (MarlinExperts.apply) instead of FlashInfer TRT-LLM (the E1 receipt ran marlin; B6/B8),
  * DS-V4-Pro geometry (hidden 7168, moe_inter 3072, 61 layers, 3 hash-routed, 384 experts, top-6),
  * EP2: each rank re-homes only its 192 local experts; the rowmap lists GLOBAL hot ids per layer (union of both ranks),
  * a VARIABLE hot count per layer (global greedy allocation of the rank's HBM row budget, done offline in the rowmap).

How the split works (no new kernels):
  stock: GEMM1 -> act -> GEMM2 writes one weighted row per (token, k) into cache3 [M*topk, K]; ops.moe_sum(cache3, out,
         topk_ids, expert_map) sums the local rows in k order.
  split: the hot call (HBM weights, expert_map_hot: global -> hot row | -1) writes the hot (token, k) rows of cache3;
         the cold call (Grace UVA weights, expert_map_cold) writes ITS rows into the SAME cache3 (marlin only writes rows
         that moe_align routed to it); then ONE stock ops.moe_sum with the original EP expert_map.
         -> same per-row products, same k-order reduction as stock.
Graph-safe: no host sync / .item() in apply; allocations are ordinary torch.empty (graph pool under capture).

Env: PIN_MODE=split|off   PIN_ROWMAP=/w/rowmap.json   PIN_SELFTEST=1 (compare split vs stock on 1 layer before freeing)
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import threading
import time

import torch

MODE = os.environ.get("PIN_MODE", "").strip().lower()
ROWMAP_PATH = os.environ.get("PIN_ROWMAP", "/w/rowmap.json")
SELFTEST = os.environ.get("PIN_SELFTEST", "1") == "1"
E_GLOBAL = 384
MIN_HOST_AVAIL = int(float(os.environ.get("PIN_MIN_HOST_GIB", "24")) * 1024**3)
MIN_HBM_FREE = int(float(os.environ.get("PIN_MIN_HBM_GIB", "4")) * 1024**3)


def _LOG(m):
    sys.stderr.write(f"PIN_HOT {time.strftime('%H:%M:%S')} {m}\n")
    sys.stderr.flush()


_STATES: dict = {}      # placeholder w1 data_ptr -> PinState
_KEEP: list = []        # placeholders + pinned host tensors backing UVA views
_orig_apply = None
_rehomed = False
_installed = False


class PinState:
    __slots__ = ("layer", "n_hot", "n_cold", "hot", "cold", "map_hot", "map_cold", "checked")

    def __init__(self, layer):
        self.layer = layer
        self.hot = {}
        self.cold = {}
        self.checked = False


def mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _uva_view(cpu: torch.Tensor) -> torch.Tensor:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    return get_accelerator_view_from_cpu_tensor(cpu)


def _placeholder(t: torch.Tensor) -> torch.Tensor:
    """Tiny HBM tensor that reports t's shape/dtype (stride-0 expand). Unique data_ptr per call."""
    base = torch.zeros([1] * t.ndim, dtype=t.dtype, device="cuda")
    _KEEP.append(base)
    return base.expand(t.shape)


def _replace_param(mod, attr, new):
    cur = getattr(mod, attr)
    if isinstance(cur, torch.nn.Parameter):
        cur.data = new
        if getattr(cur, "_vllm_is_uva_offloaded", False):
            cur._vllm_is_uva_offloaded = False
    else:
        setattr(mod, attr, new)


# --------------------------------------------------------------------------------------------------------------------
# split apply
# --------------------------------------------------------------------------------------------------------------------

def _run_part(self, hidden_states, w1, w2, s1, s2, b1, b2, topk_weights, topk_ids, emap, n_exp, activation,
              apply_router_weight_on_input, cache13, cache2, output):
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe as mm
    M, K = hidden_states.size()
    topk = topk_ids.size(1)
    Mest = math.ceil(M * n_exp / E_GLOBAL)
    for block_size_m in [8, 16, 32, 48, 64]:
        if Mest * topk / n_exp / block_size_m < 0.9:
            break
    if self.input_dtype is not None and self.input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)
    sorted_ids, expert_ids, ntpp = mm.moe_align_block_size(topk_ids, block_size_m, E_GLOBAL, emap,
                                                           ignore_invalid_experts=True)
    return mm._fused_marlin_moe(
        hidden_states=hidden_states, w1=w1, w2=w2, bias1=b1, bias2=b2, w1_scale=s1, w2_scale=s2,
        topk_weights=topk_weights, num_topk=topk, quant_type=mm.ScalarType.from_id(self.quant_type_id),
        apply_router_weight_on_input=apply_router_weight_on_input, expert_map=emap, block_size_m=block_size_m,
        sorted_token_ids=sorted_ids, expert_ids=expert_ids, num_tokens_post_padded=ntpp, activation=activation,
        activation_func=self.activation, activation_config=self.activation_config, topk_ids=topk_ids,
        input_global_scale1=self.a1_gscale, input_global_scale2=self.a2_gscale,
        global_scale1=None, global_scale2=None, w1_zeros=None, w2_zeros=None,
        intermediate_cache13=cache13, intermediate_cache2=cache2, output=output, input_dtype=self.input_dtype)


def _split_apply(self, st, output, hidden_states, topk_weights, topk_ids, activation, expert_map,
                 workspace13, workspace2, apply_router_weight_on_input):
    import vllm._custom_ops as ops
    M, K = hidden_states.size()
    topk = topk_ids.size(1)
    h, c = st.hot, st.cold
    # hot: stock workspaces (cache13 = workspace2, cache2 = workspace13, as MarlinExperts.apply swaps them)
    cache3 = _run_part(self, hidden_states, h["w1"], h["w2"], h["s1"], h["s2"], h.get("b1"), h.get("b2"),
                       topk_weights, topk_ids, st.map_hot, st.n_hot, activation, apply_router_weight_on_input,
                       workspace2, workspace13, None)
    # cold: own GEMM1 buffer (must not alias cache3), writes its rows INTO cache3
    cold13 = torch.empty((M * topk * max(2 * _inter(h), K),), device=hidden_states.device, dtype=hidden_states.dtype)
    _run_part(self, hidden_states, c["w1"], c["w2"], c["s1"], c["s2"], c.get("b1"), c.get("b2"),
              topk_weights, topk_ids, st.map_cold, st.n_cold, activation, apply_router_weight_on_input,
              cold13, workspace13, cache3)
    ops.moe_sum(cache3.view(-1, topk, K), output, topk_ids, expert_map)


def _inter(h):
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import marlin_moe_intermediate_size
    return marlin_moe_intermediate_size(h["w1"], h["w2"])


def _apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map,
           a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
    st = _STATES.get(w1.data_ptr()) if MODE == "split" else None
    if st is None or getattr(self, "_lora_context", None) is not None:
        return _orig_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                           expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                           apply_router_weight_on_input)
    if not st.checked and not torch.cuda.is_current_stream_capturing():
        _check_expert_map(st, expert_map)
    return _split_apply(self, st, output, hidden_states, topk_weights, topk_ids, activation, expert_map,
                        workspace13, workspace2, apply_router_weight_on_input)


def _check_expert_map(st, expert_map):
    """Eager-only: confirm hot∪cold maps cover exactly the rank's local experts in the EP map."""
    if expert_map is None:
        raise RuntimeError("PIN_HOT expects EP (expert_map) on this path")
    local = (expert_map >= 0)
    covered = (st.map_hot >= 0) | (st.map_cold >= 0)
    both = (st.map_hot >= 0) & (st.map_cold >= 0)
    if not torch.equal(local, covered) or bool(both.any()):
        raise RuntimeError(f"PIN_HOT layer {st.layer}: hot/cold maps do not partition the EP-local experts")
    st.checked = True


# --------------------------------------------------------------------------------------------------------------------
# re-home
# --------------------------------------------------------------------------------------------------------------------

def _load_rowmap():
    raw = json.load(open(ROWMAP_PATH))
    return {int(k): set(int(x) for x in v["hot"]) for k, v in raw["layers"].items()}, raw.get("name", ROWMAP_PATH)


def _find_routed(model):
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "RoutedExperts" and torch.is_tensor(getattr(mod, "w13_weight", None)):
            parts = name.split(".")
            li = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    li = int(parts[i + 1])
                    break
            if li is not None and ".mtp" not in name:
                out.append((li, name, mod))
    out.sort(key=lambda x: x[0])
    return out


def _find_marlin_experts(model):
    """Map id(RoutedExperts) -> MarlinExperts instance via the quant_method's modular kernel, best effort."""
    res = {}
    for _, mod in model.named_modules():
        if type(mod).__name__ != "RoutedExperts":
            continue
        qm = getattr(mod, "quant_method", None)
        found = None
        for obj in (qm, getattr(qm, "moe_kernel", None), getattr(getattr(qm, "moe_kernel", None), "impl", None)):
            if obj is None:
                continue
            fe = getattr(obj, "fused_experts", None)
            if fe is not None and type(fe).__name__ == "MarlinExperts":
                found = fe
                break
        res[id(mod)] = found
    return res


LEAD = (("w1", "w13_weight"), ("w2", "w2_weight"), ("s1", "w13_weight_scale"), ("s2", "w2_weight_scale"),
        ("b1", "w13_bias"), ("b2", "w2_bias"))


def _rehome_layer(li, routed, experts, hot_global, ep_rank, n_local, selftest):
    dev = torch.device("cuda")
    lo = ep_rank * n_local
    local_hot = sorted(g - lo for g in hot_global if lo <= g < lo + n_local)
    local_cold = sorted(set(range(n_local)) - set(local_hot))
    if not local_hot or not local_cold:
        raise RuntimeError(f"PIN_HOT layer {li}: need >=1 hot and >=1 cold row (hot={len(local_hot)})")
    st = PinState(li)
    st.n_hot, st.n_cold = len(local_hot), len(local_cold)
    mh = torch.full((E_GLOBAL,), -1, dtype=torch.int32)
    mc = torch.full((E_GLOBAL,), -1, dtype=torch.int32)
    for r, l in enumerate(local_hot):
        mh[lo + l] = r
    for r, l in enumerate(local_cold):
        mc[lo + l] = r
    st.map_hot, st.map_cold = mh.to(dev), mc.to(dev)
    src = {}
    for key, attr in LEAD:
        t = getattr(routed, attr, None)
        if torch.is_tensor(t) and t.ndim >= 1 and int(t.shape[0]) == n_local:
            src[key] = t
    for need in ("w1", "w2", "s1", "s2"):
        if need not in src:
            raise RuntimeError(f"PIN_HOT layer {li}: missing {need}; have {list(src)}")
    was_uva = bool(getattr(src["w1"], "_vllm_is_uva_offloaded", False))
    hix = torch.tensor(local_hot, dtype=torch.int64, device=dev)
    cix = torch.tensor(local_cold, dtype=torch.int64, device=dev)
    for key, t in src.items():
        st.hot[key] = t.index_select(0, hix).contiguous()                # -> HBM (reads Grace if UVA)
        cold = t.index_select(0, cix).contiguous()
        if key in ("w1", "w2"):                                           # cold weights -> pinned Grace, UVA view
            cpu = torch.empty(cold.shape, dtype=cold.dtype, pin_memory=True)
            cpu.copy_(cold)
            del cold
            _KEEP.append(cpu)
            st.cold[key] = _uva_view(cpu)
        else:                                                             # scales / bias stay in HBM
            st.cold[key] = cold
    torch.cuda.synchronize()
    if selftest and experts is not None:
        _selftest(li, experts, src, st, ep_rank, n_local)
    # swap the module params to placeholders (frees the originals) and point the quant config at the hot scales
    for key, attr in LEAD:
        if key in src:
            _replace_param(routed, attr, _placeholder(src[key]))
    qc = getattr(experts, "quant_config", None) if experts is not None else None
    if qc is not None:   # FusedMoEQuantConfig.w1_scale is a read-only property over _w1.scale
        qc._w1.scale, qc._w2.scale = st.hot["s1"], st.hot["s2"]
    src.clear()
    _STATES[routed.w13_weight.data_ptr()] = st
    return st, was_uva


def _selftest(li, experts, src, st, ep_rank, n_local):
    """Eager: random tokens routed to this rank's experts; stock apply on the original weights vs the split."""
    torch.manual_seed(1234 + li)
    dev = torch.device("cuda")
    M, topk = 12, 6
    w1, w2 = src["w1"], src["w2"]
    K = int(w1.shape[1]) * 16
    x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()
    ids = torch.randint(0, E_GLOBAL, (M, topk), device=dev, dtype=torch.int32)
    wts = torch.rand(M, topk, device=dev, dtype=torch.float32)
    emap = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev)
    lo = ep_rank * n_local
    emap[lo:lo + n_local] = torch.arange(n_local, dtype=torch.int32, device=dev)
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    act = getattr(getattr(experts, "moe_config", None), "activation", None) or MoEActivation.SILU
    N = _inter({"w1": w1, "w2": w2})
    ws13 = torch.empty((M * topk, max(N, K)), device=dev, dtype=torch.bfloat16)
    ws2 = torch.empty((M * topk * max(2 * N, K),), device=dev, dtype=torch.bfloat16)
    out_ref = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    out_new = torch.empty_like(out_ref)
    _orig_apply(experts, out_ref, x, w1, w2, wts, ids, act, E_GLOBAL, emap, None, None, ws13, ws2, None, False)
    ws13b, ws2b = torch.empty_like(ws13), torch.empty_like(ws2)
    _split_apply(experts, st, out_new, x, wts, ids, act, emap, ws13b, ws2b, False)
    torch.cuda.synchronize()
    d = (out_ref.float() - out_new.float()).abs()
    rel = (d.max() / out_ref.float().abs().max().clamp_min(1e-6)).item()
    exact = torch.equal(out_ref, out_new)
    _LOG(f"selftest layer={li} act={act} bit_exact={exact} max_abs={d.max().item():.3e} rel={rel:.3e}")
    if rel > 2e-2:
        raise RuntimeError(f"PIN_HOT selftest FAILED layer {li}: rel {rel:.3e}")


def rehome_model(model):
    global _rehomed
    from vllm.distributed import get_ep_group
    rowmap, rname = _load_rowmap()
    ep_rank = get_ep_group().rank_in_group
    routed = _find_routed(model)
    mexp = _find_marlin_experts(model)
    _LOG(f"rehome start rowmap={rname} ep_rank={ep_rank} routed_layers={len(routed)} "
         f"marlin_bound={sum(v is not None for v in mexp.values())} host_avail={mem_available_bytes()/2**30:.1f}GiB "
         f"hbm_free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB")
    hbm_first = [r for r in routed if not getattr(r[2].w13_weight, "_vllm_is_uva_offloaded", False)]
    uva_after = [r for r in routed if getattr(r[2].w13_weight, "_vllm_is_uva_offloaded", False)]
    tested = False
    hot_rows = cold_rows = 0
    for li, name, mod in hbm_first + uva_after:
        if li not in rowmap:
            _LOG(f"layer {li} not in rowmap: left stock")
            continue
        n_local = int(mod.w13_weight.shape[0])
        st, was_uva = _rehome_layer(li, mod, mexp.get(id(mod)), rowmap[li], ep_rank, n_local,
                                    SELFTEST and not tested)
        tested = True
        hot_rows += st.n_hot
        cold_rows += st.n_cold
        gc.collect()
        host, hbm = mem_available_bytes(), torch.cuda.mem_get_info()[0]
        _LOG(f"rehome layer={li} was_uva={was_uva} hot={st.n_hot} cold={st.n_cold} "
             f"host_avail={host/2**30:.1f}GiB hbm_free={hbm/2**30:.1f}GiB")
        if host < MIN_HOST_AVAIL:
            raise RuntimeError(f"PIN_HOT abort: MemAvailable {host/2**30:.1f} GiB < floor")
        if hbm < MIN_HBM_FREE:
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0] < MIN_HBM_FREE:
                raise RuntimeError(f"PIN_HOT abort: HBM free {hbm/2**30:.1f} GiB < floor")
    torch.cuda.empty_cache()
    _rehomed = True
    _LOG(f"rehome done layers={len(_STATES)} hot_rows={hot_rows} cold_rows={cold_rows} "
         f"host_avail={mem_available_bytes()/2**30:.1f}GiB hbm_free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB")


# --------------------------------------------------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------------------------------------------------

def _patch_marlin(module):
    global _orig_apply
    cls = module.MarlinExperts
    if getattr(cls.apply, "_pin_wrapped", False):
        return
    _orig_apply = cls.apply
    _apply._pin_wrapped = True
    cls.apply = _apply
    _LOG(f"patched MarlinExperts.apply mode={MODE}")


def _patch_runner(module):
    cls = getattr(module, "GPUModelRunner", None)
    if cls is None or getattr(cls.load_model, "_pin_wrapped", False):
        return
    orig = cls.load_model

    def load_model(self, *a, **k):
        orig(self, *a, **k)
        if MODE == "split" and not _rehomed:
            rehome_model(self.model)
    load_model._pin_wrapped = True
    cls.load_model = load_model
    _LOG(f"wrapped {module.__name__}.GPUModelRunner.load_model")


def install():
    global _installed
    if _installed:
        return
    import importlib.abc
    import importlib.util
    targets = {
        "vllm.model_executor.layers.fused_moe.experts.marlin_moe": _patch_marlin,
        "vllm.v1.worker.gpu.model_runner": _patch_runner,
        "vllm.v1.worker.gpu_model_runner": _patch_runner,
    }

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            p = targets.get(name)
            if p is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _o=orig_exec, _p=p):
                _o(module)
                try:
                    _p(module)
                except Exception as e:  # noqa: BLE001
                    _LOG(f"patch failed {name}: {e!r}")
                    raise
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
    for name, p in targets.items():
        if name in sys.modules:
            p(sys.modules[name])
    _installed = True
    _LOG(f"installed mode={MODE} rowmap={ROWMAP_PATH} pid={os.getpid()}")
