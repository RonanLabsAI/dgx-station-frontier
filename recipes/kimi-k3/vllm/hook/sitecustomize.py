# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""Load hook for vellum-ai/Kimi-K3-W3A16-g64 on vLLM nightly af7f9488 with --moe-backend humming and UVA expert offload.

Mounted as /opt/k3hook and put on PYTHONPATH by ../launch-k3.sh, so Python imports it at interpreter start in every
vLLM process. It imports nothing from vLLM up front: a meta-path finder patches two vLLM classes when their modules are
first imported. Set K3_REPACK3=0 to disable it.

Fix 1: 3-bit packing layout (RoutedExperts.weight_loader).
  vellum ships the routed experts in the canonical compressed-tensors `pack-quantized` int3 layout: 10 values per
  int32 (value g*10+i at bits 3i of word g, 2 pad bits), last dimension ceil(K/10) = 359 words for K=3584 and 308 for
  K=3072. vLLM's WNA16 MoE parameters and the Humming kernels expect the bit-dense layout, ceil(K*3/32) = 336 / 288
  words (32 values per 3 words; value i at bit 3i of the 96-bit little-endian stream, as in humming's
  common_pack_weight). Without this, loading fails with
  "The size of tensor a (336) must match the size of tensor b (359)".
  The hook repacks each local expert's `weight_packed` tensor on the GPU, copies the result back to the CPU, and lets
  vLLM's own loader place it (in HBM or in pinned Grace memory). Unsigned codes (+4 offset) are kept unchanged.
  On the CPU the repack took about 37 s per shard (hours for 275 shards); on the GPU about 3 s. Keeping the repacked
  tensors on the GPU grew HBM use by tens of GiB (allocator pools of the interleaved loading streams) until OOM, hence
  the copy back and an empty_cache() every 256 tensors.

Fix 2: UVA offload lost on rename (CompressedTensorsWNA16MoEMethod.process_weights_after_loading).
  Humming's MoE weight conversion replaces `w13_weight_packed` / `w2_weight_packed` with new parameters named
  `w13_weight` / `w2_weight` (delattr + setattr). vLLM's device_loading_context re-offloads replaced UVA parameters
  by NAME, so the renamed tensors silently stay in HBM; with 408 GiB/rank offloaded this ends in an OOM inside
  humming repack_weight around layer 20. The hook moves them back to pinned host memory (UVA view) after each layer.

Both are upstream bugs in this vLLM build (a canonical compressed-tensors 3-bit MoE checkpoint does not load, and UVA
offload does not survive a parameter rename); this file is a workaround, not a fix.
Credits: the packing layouts are those of compressed-tensors (neuralmagic, Apache-2.0) and humming (in vLLM,
Apache-2.0); no code is copied from either. test_k3hook.py checks the repack against both.
"""
import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = "vllm.model_executor.layers.fused_moe.routed_experts"
_TARGET2 = "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16"
_REOFF = {"n": 0, "bytes": 0}
_COUNT = {"n": 0}


def repack_10_to_dense(w):
    """[N, ceil(K/10)] int32 (10 x 3-bit per word) -> [N, K*3/32] int32 (bit-dense). K is 3584 or 3072 (Kimi K3)."""
    import torch
    n, words = w.shape
    k = words * 10
    k_true = {359: 3584, 308: 3072}.get(words, None)
    if k_true is None or k_true % 32:
        raise ValueError(f"unexpected 3-bit packed width {words}")
    shifts = torch.arange(10, dtype=torch.int32, device=w.device) * 3
    vals = ((w.unsqueeze(-1) >> shifts) & 7).reshape(n, k)[:, :k_true].to(torch.int64)
    g = vals.reshape(n, k_true // 32, 32)
    out = torch.zeros(n, k_true // 32, 3, dtype=torch.int64, device=w.device)
    for i in range(32):
        idx = 3 * i
        wi, off = idx // 32, idx % 32
        out[..., wi] |= (g[..., i] << off) & 0xFFFFFFFF
        if off + 3 > 32:
            out[..., wi + 1] |= g[..., i] >> (32 - off)
    out = out.reshape(n, k_true // 32 * 3)
    out = torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)
    return out.contiguous()


def _patch(mod):
    cls = mod.RoutedExperts
    orig = cls.weight_loader
    if getattr(orig, "_k3_repack", False):
        return

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        if ("weight_packed" in weight_name and str(loaded_weight.dtype) == "torch.int32"
                and loaded_weight.dim() == 2 and loaded_weight.shape[-1] in (359, 308)
                and getattr(getattr(self, "quant_method", None), "num_bits", None) == 3
                and self._map_global_expert_id_to_local_expert_id(expert_id) != -1):
            import torch
            loaded_weight = repack_10_to_dense(loaded_weight.to(torch.cuda.current_device())).cpu()
            _COUNT["n"] += 1
            if _COUNT["n"] % 256 == 0:
                torch.cuda.empty_cache()
            if _COUNT["n"] in (1, 1000, 100000):
                print(f"[k3hook] repacked {_COUNT['n']} 3-bit tensors (last {weight_name} -> {tuple(loaded_weight.shape)})",
                      file=sys.stderr, flush=True)
        return orig(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success)

    weight_loader._k3_repack = True
    cls.weight_loader = weight_loader
    print(f"[k3hook] patched RoutedExperts.weight_loader pid={os.getpid()}", file=sys.stderr, flush=True)


def _patch_wna16(mod):
    cls = mod.CompressedTensorsWNA16MoEMethod
    orig = cls.process_weights_after_loading
    if getattr(orig, "_k3_reoffload", False):
        return

    def process_weights_after_loading(self, layer):
        import torch
        was = {n.split("_weight")[0] for n, p in layer.named_parameters()
               if getattr(p, "_vllm_is_uva_offloaded", False) and n in ("w13_weight_packed", "w2_weight_packed")}
        orig(self, layer)
        if not was:
            return
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        for n, p in list(layer.named_parameters()):
            if n.split("_weight")[0] in was and n in ("w13_weight", "w2_weight", "w13_weight_packed", "w2_weight_packed") \
                    and p.is_cuda and not getattr(p, "_vllm_is_uva_offloaded", False):
                host = torch.empty(p.data.shape, dtype=p.data.dtype, device="cpu", pin_memory=True)
                host.copy_(p.data)
                p.data = get_accelerator_view_from_cpu_tensor(host)
                p._vllm_is_uva_offloaded = True
                _REOFF["n"] += 1
                _REOFF["bytes"] += host.numel() * host.element_size()
        torch.cuda.empty_cache()
        if _REOFF["n"] % 20 in (1, 2):
            print(f"[k3hook] re-offloaded {_REOFF['n']} humming params, {_REOFF['bytes']/2**30:.1f} GiB", file=sys.stderr, flush=True)

    process_weights_after_loading._k3_reoffload = True
    cls.process_weights_after_loading = process_weights_after_loading
    print(f"[k3hook] patched CompressedTensorsWNA16MoEMethod.process_weights_after_loading pid={os.getpid()}",
          file=sys.stderr, flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn):
        self.inner = inner
        self.fn = fn

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.fn(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fn = {_TARGET: _patch, _TARGET2: _patch_wna16}.get(name)
        if fn is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, fn)
        return spec


if os.environ.get("K3_REPACK3", "1") == "1":
    sys.meta_path.insert(0, _Finder())
