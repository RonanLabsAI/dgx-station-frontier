# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""k3ds_guard -- the DSpark guard from our DS-V4-Pro run (R3.1b: failfast, telemetry, acceptance-gated K; R3.1c:
skip-draft; receipts logs/dsv4-pro-two-stations/dspark-guard-r31b.log and dspark-skipdraft-r31c.log), ported to the
Kimi K3 DSpark drafter (Inferact/Kimi-K3-DSpark, K3DSparkForCausalLM) on vLLM nightly af7f9488 (V2 runner, TP2+EP2
across two GB300 Stations). Use: recipes/kimi-k3 README section A2.

Loaded by hook/sitecustomize.py only when K3DS_GUARD=1. EVERY part below is off unless its env var turns it on, so an
import with no env set patches nothing.

What carried over unchanged from R3.1b/R3.1c (af7f9488 is the same vLLM commit, and DSparkSpeculator inherits
DFlashSpeculator.propose there, so the worker-side wrappers apply to the K3 drafter as they did to V4-Pro's):
  1. FAILFAST (R31B_FAILFAST=1): an exception in execute_model / sample_tokens / execute_dummy_batch on ANY rank exits
     the worker (code 86) after the output rank has reported it. A swallowed non-output-rank exception (the R3.1 hang
     root cause) becomes a loud, attributable failure. Pair with VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120 (launcher).
  3. TELEMETRY (K3DS_STATS=path): mem_get_info free bytes (min / last) every 5 s plus the guard counters, per rank;
     the K controller's state goes to <path>.kctl.json from the EngineCore process.
  5. ACCEPTANCE-GATED K (R31B_KCTL=1; EngineCore only): the scheduler's dense dynamic-SD lookup is wrapped by a
     controller that schedules K_OFF while the EMA acceptance length is low, with probes and exponential back-off.
     K is decided in the scheduler and broadcast in SchedulerOutput, so every rank sees the same K (vllm#54011 is the
     rank-local counter-example). Thresholds R31B_KCTL_OFF / R31B_KCTL_ON (this recipe sets 1.6 / 2.0 around K3's break-even
     acceptance of ~1.78 at K=3; V4-Pro used 1.5 / 1.9 around 1.68).
  6. SKIP-DRAFT (R31C_SKIPDRAFT=1): when the broadcast SchedulerOutput says this step's drafts will be verified at K=0,
     propose() runs a light step (combine_hidden_states + per-group prepare_dflash_inputs + eager context K/V of every
     target row; no draft forward, no collective). Rank agreement: every R31B_CHECK_EVERY proposals (alias
     R31C_CHECK_EVERY) the (calls, light) counters are all-reduced over the TP CPU group; a mismatch is FAILFAST.

What changed for K3 (and why):
  2. DRAFTER EAGER-PATH WARM-UP (K3DS_DRAFT_WARMUP=relax|full|skip) replaces R3.1b's drafter DeepGEMM warm-up. The K3
     drafter is BF16 (no fp8 DeepGEMM linears, so R3.1b's warm-up would find nothing to warm), but its eager path is
     the same R3.1b risk class: every mixed/prefill step runs combine_hidden_states (context_proj, a BF16 GEMM of
     M x 35,840 -> 7,168) and precompute_and_store_context_kv (context_kv_proj, a BF16 GEMM M x 7,168 -> 5 x 576, a
     grouped RMSNorm, RoPE and concat_and_cache_mla_grouped) eagerly, at M values decode traffic never produces.
     Any lazily loaded kernel (cuBLAS/cuBLASLt heuristics + module load, the vLLM CUDA ops) would then load in the
     serving hot path. The warm-up runs exactly those two calls over an M set after vLLM's kernel_warmup, with every
     context slot = PAD_SLOT_ID (-1), which concat_and_cache_mla_grouped skips (csrc cache_kernels.cu, slot_idx < 0),
     so no KV block is written. All of it is rank-local (ReplicatedLinear / disable_tp=True, local insert), so it
     cannot desynchronize collectives. "relax" = M in 1..64, every 16 to 512, every 128 to max_num_batched_tokens,
     and max_num_batched_tokens itself; "full" = every M in 1..max_num_batched_tokens.
  7. ZERO-TIER DRAFTER GRAPHS (K3DS_ZERO_TIER=1): the base schedule [[1,1,3],[2,3,2],[4,16,0]] puts K=0 in the dense
     lookup itself (R3.1c only ever appended K_OFF=0). cudagraph_utils turns each scheduled K into a decode query
     length k + (decode_query_len - num_speculative_tokens); for the DSpark drafter's manager that offset is 0, so
     K=0 becomes query length 0 and round_up(n, 0) raises ZeroDivisionError at capture (the R3.1b B2 first-boot
     failure). R3.1c's patch returns early when 0 is already in the list, so it does not cover this. The drafter never
     runs at K=0 under SKIPDRAFT (the step is a light step), so for that one caller the 0 entries are replaced by the
     largest scheduled K; the K SET the drafter captures is then the schedule's non-zero Ks. Target managers keep 0
     (query length 1, a plain decode graph).
  Dropped: R3.1b fault injection (R31B_INJECT) -- not included here.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time
import traceback

FAILFAST = os.environ.get("R31B_FAILFAST", "0") == "1"
FAILFAST_METHODS = {"execute_model", "sample_tokens", "execute_dummy_batch"}
FAILFAST_CODE = 86
DRAFT_WARMUP = os.environ.get("K3DS_DRAFT_WARMUP", "skip").strip().lower()
STATS_PATH = os.environ.get("K3DS_STATS", "")
KCTL = os.environ.get("R31B_KCTL", "0") == "1"
SKIPDRAFT = os.environ.get("R31C_SKIPDRAFT", "0") == "1"
CHECK_EVERY = int(os.environ.get("R31B_CHECK_EVERY", os.environ.get("R31C_CHECK_EVERY", "0")) or 0)
ZERO_TIER = os.environ.get("K3DS_ZERO_TIER", "0") == "1"
PAD_SLOT_ID = -1  # vllm/v1/attention/backends/utils.py at af7f9488

_log = lambda m: (sys.stderr.write(f"K3DS {m}\n"), sys.stderr.flush())  # noqa: E731
_state = {"failfast": 0, "proposals": 0, "draft_warm_calls": 0, "draft_warm_s": None, "draft_warm_m": None,
          "mem_free_min": None, "mem_free_last": None, "mem_total": None, "kctl": None}


# ---------------------------------------------------------------------------------------------- 1. failfast
def _fatal(method: str, rank, exc: BaseException) -> None:
    _state["failfast"] += 1
    _log(f"FAILFAST rank={rank} method={method}: {exc!r}\n" + "".join(traceback.format_exception(exc)))
    try:
        _dump_stats()
    except Exception:
        pass
    os._exit(FAILFAST_CODE)


def make_execute_worker_rpc(orig, cloudpickle_loads, partial):
    """Replacement for WorkerProc._execute_worker_rpc (same contract as af7f9488) that never lets a rank swallow an
    exception from a collective-bearing method and carry on with the next RPC (unchanged from R3.1b)."""

    def _execute_worker_rpc(self, rpc_request):
        method, args, kwargs, output_rank = rpc_request
        name = method if isinstance(method, str) else "<pickled>"
        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)
            else:
                func = partial(cloudpickle_loads(method), self.worker)
            output = func(*args, **kwargs)
            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)
        except Exception as e:
            if FAILFAST and name in FAILFAST_METHODS:
                if output_rank is None or self.rank == output_rank:
                    try:
                        self.handle_output(e)
                        time.sleep(0.5)  # async scheduling hands the output to a sender thread
                    except Exception:
                        pass
                _fatal(name, getattr(self, "rank", "?"), e)
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            _log(f"WorkerProc exception (not failfast: method={name}): {e!r}\n" + traceback.format_exc())
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)

    _execute_worker_rpc._k3ds = True
    _execute_worker_rpc._k3ds_orig = orig
    return _execute_worker_rpc


def _patch_executor(module):
    if not FAILFAST:
        return
    cls = module.WorkerProc
    if getattr(cls._execute_worker_rpc, "_k3ds", False):
        return
    cls._execute_worker_rpc = make_execute_worker_rpc(cls._execute_worker_rpc, module.cloudpickle.loads,
                                                      module.partial)
    _log(f"patched WorkerProc._execute_worker_rpc (failfast=True, methods={sorted(FAILFAST_METHODS)})")


# -------------------------------------------------------------------------- 2. drafter eager-path warm-up (K3)
def draft_m_values(mode: str, max_tokens: int) -> list[int]:
    if mode == "full":
        return list(range(1, max_tokens + 1))
    ms = set(range(1, min(64, max_tokens) + 1))
    ms.update(range(16, min(512, max_tokens) + 1, 16))
    ms.update(range(128, max_tokens + 1, 128))
    ms.add(max_tokens)
    return sorted(ms)


def warm_drafter_eager(worker) -> None:
    """Run the K3 drafter's two eager context calls over draft_m_values(), every slot PAD (no cache write)."""
    if DRAFT_WARMUP not in ("relax", "full"):
        return
    runner = getattr(worker, "model_runner", None)
    spec = getattr(runner, "speculator", None)
    model = getattr(spec, "model", None)
    if model is None or not hasattr(model, "precompute_and_store_context_kv") \
            or not hasattr(model, "combine_hidden_states"):
        _log("drafter warm-up: no DSpark draft model with an eager context path (nothing to do)")
        return
    import torch
    cfg = model.config
    width = int(cfg.target_hidden_size) * int(cfg.num_target_layers)
    max_tokens = int(worker.scheduler_config.max_num_batched_tokens)
    p = next(model.parameters())
    ms = draft_m_values(DRAFT_WARMUP, max_tokens)
    t0 = time.perf_counter()
    with torch.inference_mode():
        x = torch.zeros((max_tokens, width), device=p.device, dtype=p.dtype)
        pos = torch.zeros((max_tokens,), device=p.device, dtype=torch.int64)
        slots = torch.full((max_tokens,), PAD_SLOT_ID, device=p.device, dtype=torch.int64)
        for m in ms:
            h = model.combine_hidden_states(x[:m])
            model.precompute_and_store_context_kv(h, pos[:m], slots[:m])
        del x, pos, slots
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    _state["draft_warm_calls"], _state["draft_warm_s"], _state["draft_warm_m"] = 2 * len(ms), round(dt, 2), \
        [ms[0], ms[-1], len(ms)]
    free_s = ""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_s = f"; free after {free / 2**20:.0f} MiB of {total / 2**30:.1f} GiB"
    _log(f"drafter eager warm-up ({DRAFT_WARMUP}): {len(ms)} M values {ms[0]}..{ms[-1]}, width {width}, "
         f"{2 * len(ms)} calls, {dt:.1f} s{free_s}")


def _patch_gpu_worker(module):
    if DRAFT_WARMUP not in ("relax", "full") and not STATS_PATH:
        return
    orig = module.kernel_warmup
    if getattr(orig, "_k3ds", False):
        return

    def kernel_warmup(worker, *args, **kwargs):
        res = orig(worker, *args, **kwargs)
        if kwargs.get("process_local_only"):
            return res
        warm_drafter_eager(worker)  # an exception here fails the boot loudly, which is what we want
        _start_stats()
        return res

    kernel_warmup._k3ds = True
    module.kernel_warmup = kernel_warmup
    _log(f"patched gpu_worker.kernel_warmup (drafter eager warm-up={DRAFT_WARMUP}, stats={bool(STATS_PATH)})")


# ------------------------------------------------------------------------------------------- 3. telemetry
_stats_started = False


def _dump_stats() -> None:
    if not STATS_PATH:
        return
    tmp = STATS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"time": time.time(), **_state, "skip": dict(_skip)}, f)
    os.replace(tmp, STATS_PATH)


def _start_kctl_stats(ctl) -> None:
    if not STATS_PATH:
        return
    path = STATS_PATH.rsplit(".json", 1)[0] + ".kctl.json"

    def loop():
        while True:
            try:
                with open(path + ".tmp", "w") as f:
                    json.dump({"time": time.time(), **ctl.snapshot()}, f)
                os.replace(path + ".tmp", path)
            except Exception as exc:
                _log(f"kctl stats failed: {exc!r}")
            time.sleep(5)

    threading.Thread(target=loop, daemon=True, name="k3ds-kctl-stats").start()


def _start_stats() -> None:
    global _stats_started
    if _stats_started or not STATS_PATH:
        return
    _stats_started = True
    import torch

    def loop():
        while True:
            try:
                free, total = torch.cuda.mem_get_info()
                _state["mem_free_last"], _state["mem_total"] = free, total
                if _state["mem_free_min"] is None or free < _state["mem_free_min"]:
                    _state["mem_free_min"] = free
                _dump_stats()
            except Exception as exc:
                _log(f"stats failed: {exc!r}")
            time.sleep(5)

    threading.Thread(target=loop, daemon=True, name="k3ds-stats").start()


# --------------------------------------------------------------------------- 6. skip-draft (R3.1c, worker side)
_skip = {"next_k": None, "calls": 0, "light": 0, "full": 0, "checks": 0, "async": None}

PROPOSE_PARAMS = ("input_batch", "attn_metadata", "slot_mappings", "last_hidden_states", "aux_hidden_states",
                  "num_sampled", "num_rejected", "last_sampled", "next_prefill_tokens", "temperature", "seeds")


def record_next_k(scheduler_output, dummy_run: bool) -> None:
    """Called at the top of GPUModelRunner.execute_model with the step's broadcast SchedulerOutput."""
    k = None if dummy_run else getattr(scheduler_output, "num_spec_tokens_to_schedule", None)
    _skip["next_k"] = None if k is None else int(k)


def light_step(next_k, dummy_run: bool, is_profile: bool, async_sched: bool) -> bool:
    return bool(SKIPDRAFT and async_sched and not dummy_run and not is_profile and next_k == 0)


def light_propose(spec, prepare_dflash_inputs, a: dict):
    """DFlashSpeculator.propose (af7f9488) minus the draft forward (unchanged from R3.1c): same hidden-state staging,
    same per-group prepare_dflash_inputs call, then the context K/V precompute for ALL target rows eagerly."""
    import torch
    input_batch = a["input_batch"]
    num_reqs = input_batch.num_reqs
    num_target_tokens = input_batch.num_tokens
    with torch.inference_mode():
        aux = a.get("aux_hidden_states")
        if aux:
            hidden_states = spec.model.combine_hidden_states(torch.cat(aux, dim=-1))
        else:
            hidden_states = a["last_hidden_states"]
        spec.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])
        spec.prepare_context_anchor(input_batch, a["num_rejected"])
        if spec.pcp_manager is not None:
            spec.block_tables.gather_block_tables(input_batch.idx_mapping, num_reqs_padded=num_reqs)
        for i, gid in enumerate(spec.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                spec.input_buffers,
                spec.block_tables.slot_mappings[gid],
                spec.context_positions,
                spec._context_slot_mappings[i],
                spec.sample_indices,
                spec.sample_pos,
                spec.sample_idx_mapping,
                spec.temperature,
                spec.seeds,
                input_batch,
                a["num_sampled"],
                a["num_rejected"],
                a["last_sampled"],
                a["next_prefill_tokens"],
                a["temperature"],
                a["seeds"],
                spec.block_tables.input_block_tables[gid],
                spec.block_tables.kernel_block_sizes[gid],
                spec.block_tables.cp_rank,
                spec.dcp_size,
                spec.block_tables.cp_interleave,
                spec.parallel_drafting_token_id,
                spec.num_query_per_req,
                spec.num_speculative_steps,
                spec.max_num_reqs,
                spec.max_num_tokens,
                spec.max_model_len,
                spec.sample_from_anchor,
            )
        spec._precompute_context_kv(0, num_target_tokens)
    # Never verified: the next step schedules 0 placeholders for every request of this step.
    return spec.draft_tokens[:num_reqs]


def _rank_check(n_every: int) -> None:
    """Every n_every proposals, all-reduce (calls, light) over the TP CPU group and die if the ranks disagree."""
    if n_every <= 0 or _skip["calls"] % n_every:
        return
    import torch
    import torch.distributed as dist
    group, rank, world = _tp_cpu_group()
    if world <= 1:
        return
    c, s = _skip["calls"], _skip["light"]
    t = torch.tensor([c, s, -c, -s], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    _skip["checks"] += 1
    if not (t[0] == -t[2] and t[1] == -t[3]):
        _skip["check_failed"] = True
        _fatal("skipdraft-rank-check", rank,
               RuntimeError(f"TP ranks disagree on light steps: calls max {t[0]} min {-t[2]}, "
                            f"light max {t[1]} min {-t[3]}"))


def _tp_cpu_group():
    from vllm.distributed import get_tp_group
    tp = get_tp_group()
    return tp.cpu_group, tp.rank_in_group, tp.world_size


def make_skip_propose(orig, prepare_dflash_inputs, check_every: int = 0):
    def propose(self, *args, **kwargs):
        a = dict(zip(PROPOSE_PARAMS, args))
        a.update(kwargs)
        dummy, profile = bool(a.get("dummy_run", False)), bool(a.get("is_profile", False))
        if dummy or profile:
            return orig(self, *args, **kwargs)
        if _skip["async"] is None:
            _skip["async"] = bool(getattr(self.vllm_config.scheduler_config, "async_scheduling", False))
            if not _skip["async"]:
                _log("SKIPDRAFT: async scheduling is off, so step N's K does not bound step N+1's drafts; disabled")
        _skip["calls"] += 1
        _state["proposals"] += 1
        if light_step(_skip["next_k"], dummy, profile, _skip["async"]):
            _skip["light"] += 1
            out = light_propose(self, prepare_dflash_inputs, a)
        else:
            _skip["full"] += 1
            out = orig(self, *args, **kwargs)
        _rank_check(check_every)
        return out

    propose._k3ds = True
    return propose


def _patch_dflash(module):
    if not SKIPDRAFT:
        return
    cls = module.DFlashSpeculator
    if getattr(cls.propose, "_k3ds", False):
        return
    cls.propose = make_skip_propose(cls.propose, module.prepare_dflash_inputs, CHECK_EVERY)
    _log(f"patched DFlashSpeculator.propose (skip-draft light steps at K=0, rank check every {CHECK_EVERY})")


def _patch_model_runner(module):
    if not SKIPDRAFT:
        return
    cls = module.GPUModelRunner
    orig = cls.execute_model
    if getattr(orig, "_k3ds", False):
        return

    def execute_model(self, scheduler_output, *args, **kwargs):
        record_next_k(scheduler_output, bool(kwargs.get("dummy_run", args[1] if len(args) > 1 else False)))
        return orig(self, scheduler_output, *args, **kwargs)

    execute_model._k3ds = True
    cls.execute_model = execute_model
    _log("patched GPUModelRunner.execute_model (records the broadcast num_spec_tokens_to_schedule)")


# ------------------------------------------------------------------------------ 5. acceptance-gated K
class AcceptanceGatedK:
    """Scheduler-side K choice (unchanged from R3.1c). lookup[batch_size] -> K, like vLLM's dense dynamic-SD table.
    A batch size whose base K is 0 (here: >= 4 running) always gets 0: no probes there, nothing to gate."""

    def __init__(self, base_lookup: list[int], k_off: int = 0, off_below: float = 1.5, on_above: float = 1.9,
                 alpha: float = 0.2, probe_every: int = 16, min_obs: int = 4, probe_max: int | None = None):
        self.base = list(base_lookup)
        self.k_off = k_off
        self.off_below, self.on_above = off_below, on_above
        self.alpha, self.probe_every, self.min_obs = alpha, probe_every, min_obs
        self.probe_max = probe_every if probe_max is None else max(probe_every, probe_max)
        self.probe_gap = probe_every
        self._since_probe = self._probe_left = 0
        self.ema = None
        self.on = True
        self.steps_off = 0
        self._acc = self._n = 0
        self.flips = 0
        self.probes = 0
        self.k_hist: dict[int, int] = {}

    def observe(self, num_draft_tokens: int, num_accepted: int) -> None:
        if num_draft_tokens > 0:
            self._acc += num_accepted
            self._n += 1

    def _fold(self) -> None:
        if not (self._n >= self.min_obs or (self._n > 0 and self.ema is None)):
            return
        al = 1.0 + self._acc / self._n
        self._acc = self._n = 0
        if not self.on and self.k_off == 0:
            self.ema = al
            if al > self.on_above:
                self.on = True
                self.flips += 1
            else:
                self.probe_gap = min(self.probe_gap * 2, self.probe_max)
            return
        self.ema = al if self.ema is None else (1 - self.alpha) * self.ema + self.alpha * al
        if self.on and self.ema < self.off_below:
            self.on, self.steps_off = False, 0
            self.probe_gap, self._since_probe, self._probe_left = self.probe_every, 0, 0
            self.flips += 1
        elif not self.on and self.ema > self.on_above:
            self.on = True
            self.flips += 1

    def _choose(self, batch_size: int) -> int:
        self._fold()
        k = self.base[batch_size]
        if self.on or k == 0:
            return k
        self.steps_off += 1
        if self._probe_left > 0:
            self._probe_left -= 1
            return k
        self._since_probe += 1
        if self._since_probe >= self.probe_gap:
            self._since_probe = 0
            self.probes += 1
            if self.k_off == 0:
                self._probe_left = max(1, -(-self.min_obs // max(1, batch_size))) - 1
            return k
        return self.k_off

    def __getitem__(self, batch_size: int) -> int:
        k = self._choose(batch_size)
        self.k_hist[k] = self.k_hist.get(k, 0) + 1
        return k

    def snapshot(self) -> dict:
        return {"on": self.on, "ema": None if self.ema is None else round(self.ema, 3), "flips": self.flips,
                "steps_off": self.steps_off, "probes": self.probes, "probe_gap": self.probe_gap,
                "k_hist": dict(self.k_hist)}


def _patch_scheduler(module):
    if not KCTL:
        return
    cls = module.Scheduler
    if getattr(cls.__init__, "_k3ds", False):
        return
    orig_init, orig_stats = cls.__init__, cls.make_spec_decoding_stats

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if self.dynamic_sd_lookup is None:
            _log("KCTL: no num_speculative_tokens_per_batch_size in --speculative-config; controller NOT installed")
            return
        ctl = AcceptanceGatedK(
            self.dynamic_sd_lookup,
            k_off=kctl_k_off(),
            off_below=float(os.environ.get("R31B_KCTL_OFF", "1.5")),
            on_above=float(os.environ.get("R31B_KCTL_ON", "1.9")),
            probe_every=int(os.environ.get("R31B_KCTL_PROBE", "16")),
            probe_max=int(os.environ.get("R31C_PROBE_MAX", "256")) if SKIPDRAFT else None)
        self.dynamic_sd_lookup = ctl
        _state["kctl"] = ctl.snapshot()
        _start_kctl_stats(ctl)
        _log(f"KCTL installed: base={ctl.base[1:]} k_off={ctl.k_off} off<{ctl.off_below} on>{ctl.on_above} "
             f"probe/{ctl.probe_every}..{ctl.probe_max} skipdraft={SKIPDRAFT}")

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                 num_invalid_spec_tokens, request_id):
        ctl = self.dynamic_sd_lookup
        if isinstance(ctl, AcceptanceGatedK):
            n = num_draft_tokens - ((num_invalid_spec_tokens or {}).get(request_id, 0))
            ctl.observe(n, num_accepted_tokens)
        return orig_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                          num_invalid_spec_tokens, request_id)

    __init__._k3ds = True
    cls.__init__ = __init__
    cls.make_spec_decoding_stats = make_spec_decoding_stats
    _log("patched Scheduler (acceptance-gated K)")


# ----------------------------------------------------------------- 5+7. CUDA-graph K sets (target / drafter)
def _caller_new_tokens_per_step(frame) -> int | None:
    """num_new_sampled_tokens_per_step of the CudaGraphManager whose _init_candidates is calling (None if unknown).
    Same formula as af7f9488 cudagraph_utils: decode_query_len - vllm_config.num_speculative_tokens."""
    mgr = frame.f_locals.get("self") if frame is not None else None
    try:
        return int(mgr.decode_query_len) - int(mgr.vllm_config.num_speculative_tokens)
    except Exception:
        return None


def drafter_dense(dense: list[int]) -> list[int]:
    """Zero-tier fix: the drafter's dense K list with every 0 replaced by the largest scheduled K (never 0)."""
    kmax = max(dense[1:], default=0)
    if kmax <= 0:
        return dense
    return dense[:1] + [k if k > 0 else kmax for k in dense[1:]]


def _patch_cudagraph_utils(module):
    """K3DS_ZERO_TIER: no query length 0 for the drafter. KCTL: target decode graphs for K_OFF too (R3.1c)."""
    if not (ZERO_TIER or KCTL):
        return
    orig = module.build_dynamic_sd_schedule_lookup
    if getattr(orig, "_k3ds", False):
        return
    k_off = kctl_k_off() if KCTL else None

    def build_dynamic_sd_schedule_lookup(*args, **kwargs):
        dense = orig(*args, **kwargs)
        new_tokens = _caller_new_tokens_per_step(sys._getframe(1))
        if new_tokens == 0:
            # DSpark drafter's manager (decode_query_len == num_speculative_tokens): never a query length of 0.
            return drafter_dense(dense) if (ZERO_TIER or k_off == 0) else dense
        if k_off is not None and k_off not in dense[1:]:
            return dense + [k_off]
        return dense

    build_dynamic_sd_schedule_lookup._k3ds = True
    module.build_dynamic_sd_schedule_lookup = build_dynamic_sd_schedule_lookup
    _log(f"patched cudagraph_utils.build_dynamic_sd_schedule_lookup (zero_tier={ZERO_TIER}, "
         f"k_off={k_off}{', target only' if k_off == 0 else ''})")


def kctl_k_off() -> int:
    """K the controller schedules while OFF: 0 only with SKIPDRAFT (light step); otherwise clamped to >= 1."""
    k_off = int(os.environ.get("R31B_KCTL_KOFF", "0"))
    if k_off < 1 and not SKIPDRAFT:
        _log(f"KCTL: K_OFF={k_off} unsupported with DSpark graphs unless R31C_SKIPDRAFT=1; using 1")
        k_off = 1
        os.environ["R31B_KCTL_KOFF"] = "1"
    return max(0, k_off)


# ----------------------------------------------------------------------------------------------- install
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


TARGETS = {
    "vllm.v1.worker.gpu.cudagraph_utils": _patch_cudagraph_utils,
    "vllm.v1.executor.multiproc_executor": _patch_executor,
    "vllm.v1.worker.gpu_worker": _patch_gpu_worker,
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator": _patch_dflash,
    "vllm.v1.worker.gpu.model_runner": _patch_model_runner,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
}


def enabled() -> dict:
    return {"failfast": FAILFAST, "draft_warmup": DRAFT_WARMUP, "stats": bool(STATS_PATH), "kctl": KCTL,
            "skipdraft": SKIPDRAFT, "check_every": CHECK_EVERY, "zero_tier": ZERO_TIER}


def install():
    on = enabled()
    if not (FAILFAST or DRAFT_WARMUP in ("relax", "full") or STATS_PATH or KCTL or SKIPDRAFT or ZERO_TIER):
        _log("installing: every part off (no K3DS_*/R31B_*/R31C_* flag set); nothing patched")
        return
    _log(f"installing: {json.dumps(on, sort_keys=True)} pid={os.getpid()}")
    for target, patch in TARGETS.items():
        if target in sys.modules:
            patch(sys.modules[target])
        else:
            sys.meta_path.insert(0, _Finder(target, patch))
