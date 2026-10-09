# Derived from original-el8/dgx-station-gb300-research @ 7699ae54, deepseek-v4.1-flash/m3/hook/peer_tier2.py
# Copyright 2026 Jason Cook. Licensed under the Apache License 2.0.
# Modifications Copyright 2026 RonanLabs (Apache License 2.0): DeepSeek-V4-Pro geometry (hidden 7,168), BF16 rows
# (no MXFP8 on the wire, so the sidecar can run W4A16 and match vLLM's marlin path), strided input/output rows,
# a handshake word plus a per-layer warm-count table so the GB300 refuses a sidecar built from another rowmap.
"""Shared-memory protocol between one GB300 vLLM worker and its RTX PRO 6000 warm-expert sidecar.

Per hooked MoE layer the GB300:

1. packs only the tokens that have a warm route (BF16 activations, local warm ids, route weights) into a pinned
   shared host buffer and publishes the row count with a sequence number (release, system scope);
2. runs its own experts (HBM + Grace) with the warm experts masked out of its expert map;
3. waits (acquire, system scope) for the sidecar to finish that sequence, then adds the sidecar's rows back.

Every GB300 step is a device kernel with no host sync, so vLLM's CUDA graphs capture it. A layer with no warm rows
skips the wait (the sidecar still acknowledges the sequence). The wait has a timeout: a missing sidecar drops the
warm contribution and raises a counter (words[2]); E3 requires that counter to stay 0.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os

import torch
import triton
import triton.language as tl

HIDDEN = 7168
TOPK = 6
MAX_ROWS = int(os.environ.get("E3_MAX_ROWS", "8192"))
# Buckets up to this size are copied whole inside the sidecar's CUDA graphs (0 disables; both sides must agree).
SMALL_ROWS = int(os.environ.get("E3_SMALL_ROWS", "64"))
MAX_LAYERS = 64
MAGIC = 0x45335F56345F5052  # "E3_V4_PR", written by the sidecar once it is serving

# Byte layout. Global words (int64) at 0: [0] published seq, [1] completed seq, [2] timeouts, [3] rows sent,
# [4] tokens seen, [5] ns the GB300 spent waiting, [6] waits, [7] MAGIC when serving, [8] rowmap hash,
# [9] sidecar pid, [10] ep rank served. int32 warm-count table at TABLE_OFF (one entry per decoder layer).
TABLE_OFF = 1024
SLOT = 4096          # header: [0] seq, [1] layer, [2] rows, [3] tokens
HEADER = 64
X_OFF = 8192
IDS_OFF = X_OFF + MAX_ROWS * HIDDEN * 2
W_OFF = IDS_OFF + MAX_ROWS * TOPK * 4
OUT_OFF = W_OFF + MAX_ROWS * TOPK * 4
TOTAL_BYTES = (OUT_OFF + MAX_ROWS * HIDDEN * 2 + 4095) // 4096 * 4096
PATH = os.environ.get("E3_SHM", "/dev/shm/e3_dsv4pro_peer")


def load_rowmap(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return raw


def rank_warm(rowmap: dict, ep_rank: int, n_local: int) -> dict[int, list[int]]:
    """Global warm expert ids per layer that EP rank `ep_rank` owns under linear placement, sorted.

    The sorted position of an id in this list is its local warm index on both sides of the protocol.
    """
    lo, hi = ep_rank * n_local, (ep_rank + 1) * n_local
    out = {}
    for key, v in rowmap["layers"].items():
        ids = sorted(int(e) for e in v.get("warm", []) if lo <= int(e) < hi)
        if ids:
            out[int(key)] = ids
    return out


def warm_hash(warm: dict[int, list[int]]) -> int:
    blob = json.dumps({str(k): warm[k] for k in sorted(warm)}, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little", signed=True)


def open_shared(create: bool) -> mmap.mmap:
    # Open an existing buffer without O_CREAT first: fs.protected_regular refuses O_CREAT on another user's file in
    # sticky /dev/shm, even for root.
    try:
        fd = os.open(PATH, os.O_RDWR)
    except FileNotFoundError:
        if not create:
            raise
        fd = os.open(PATH, os.O_RDWR | os.O_CREAT, 0o666)
    if os.fstat(fd).st_size < TOTAL_BYTES:
        os.ftruncate(fd, TOTAL_BYTES)
    buf = mmap.mmap(fd, TOTAL_BYTES)
    os.close(fd)
    return buf


def register(buf: mmap.mmap, device: torch.device) -> tuple[torch.Tensor, int]:
    """Pin the mapping in this process's CUDA context; return (host view, device pointer)."""
    from cuda.bindings import runtime as rt

    host = torch.frombuffer(buf, dtype=torch.uint8)
    pointer = host.data_ptr()
    flags = rt.cudaHostRegisterMapped | rt.cudaHostRegisterPortable
    with torch.cuda.device(device):
        (error,) = rt.cudaHostRegister(pointer, TOTAL_BYTES, flags)
        if error != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostRegister failed: {error}")
        error, dptr = rt.cudaHostGetDevicePointer(pointer, 0)
        if error != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {error}")
    return host, int(dptr)


class _Bytes:
    def __init__(self, pointer: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (pointer, False), "version": 3,
        }


def device_view(dptr: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(_Bytes(dptr, TOTAL_BYTES), device=device)


@triton.jit
def _pack(X, SX, IDS, WTS, HAS, POS, PX, PIDS, PW,
          H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr):
    """Copy token t's BF16 row (row stride SX) to shared row POS[t] if it has a warm route."""
    t = tl.program_id(0).to(tl.int64)
    if tl.load(HAS + t) == 0:
        return
    r = tl.load(POS + t).to(tl.int64)
    for off in tl.static_range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < H
        tl.store(PX + r * H + cols, tl.load(X + t * SX + cols, mask=mask), mask=mask)
    k = tl.arange(0, BLOCK_K)
    tl.store(PIDS + r * K + k, tl.load(IDS + t * K + k, mask=k < K), mask=k < K)
    tl.store(PW + r * K + k, tl.load(WTS + t * K + k, mask=k < K), mask=k < K)


@triton.jit(do_not_specialize=["LAYER", "T"])
def _publish(WORDS, HEADER_PTR, SEQ, LAYER, ROWS, T, PIDS, K: tl.constexpr, PAD: tl.constexpr):
    """Mask unused rows below PAD, advance the device sequence, write the header, publish with release.

    The sidecar copies whole buckets of up to PAD rows inside its CUDA graphs, so rows past ROWS must carry masked
    routes; otherwise stale ids make it read experts no live row routes to.
    """
    rows = tl.load(ROWS).to(tl.int64)
    if PAD > 0:
        lane = tl.arange(0, 512)
        row, col = lane // 8, lane % 8
        tl.store(PIDS + row * K + col, tl.full((512,), -1, tl.int32),
                 mask=(row >= rows) & (row < PAD) & (col < K))
        tl.debug_barrier()
    seq = tl.load(SEQ) + 1
    tl.store(SEQ, seq)
    tl.store(HEADER_PTR + 1, LAYER.to(tl.int64))
    tl.store(HEADER_PTR + 2, rows)
    tl.store(HEADER_PTR + 3, T.to(tl.int64))
    tl.store(HEADER_PTR + 0, seq)
    tl.atomic_add(WORDS + 3, rows, sem="relaxed", scope="sys")
    tl.atomic_add(WORDS + 4, T.to(tl.int64), sem="relaxed", scope="sys")
    tl.atomic_xchg(WORDS, seq, sem="release", scope="sys")


@triton.jit
def _globaltimer(dep):
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,l", [dep], dtype=tl.int64,
                                     is_pure=False, pack=1)


@triton.jit
def _wait(WORDS, SEQ, ROWS, OK, TIMEOUT: tl.constexpr):
    """Spin (acquire) until the sidecar completed this sequence; OK=0 on timeout or no rows.

    Accumulates the spin time (ns, GPU globaltimer) in WORDS[5] and the number of waits in WORDS[6], so the stall
    on the sidecar can be measured under CUDA graphs.
    """
    if tl.load(ROWS) == 0:
        tl.store(OK, 0)
        return
    seq = tl.load(SEQ)
    t0 = _globaltimer(seq)
    spins = 0
    done = tl.atomic_add(WORDS + 1, 0, sem="acquire", scope="sys")
    while (done < seq) & (spins < TIMEOUT):
        done = tl.atomic_add(WORDS + 1, 0, sem="acquire", scope="sys")
        spins += 1
    t1 = _globaltimer(done)
    tl.atomic_add(WORDS + 5, t1 - t0, sem="relaxed", scope="sys")
    tl.atomic_add(WORDS + 6, 1, sem="relaxed", scope="sys")
    if done < seq:
        tl.atomic_add(WORDS + 2, 1, sem="relaxed", scope="sys")
        tl.store(OK, 0)
    else:
        tl.store(OK, 1)


@triton.jit
def _scatter_add(Y, SY, OUT, HAS, POS, OK, H: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    if (tl.load(OK) == 0) | (tl.load(HAS + t) == 0):
        return
    r = tl.load(POS + t).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    y = tl.load(Y + t * SY + cols, mask=mask).to(tl.float32)
    o = tl.load(OUT + r * H + cols, mask=mask).to(tl.float32)
    tl.store(Y + t * SY + cols, (y + o).to(Y.dtype.element_ty), mask=mask)


class PeerTierV4:
    """GB300 side of the protocol; one instance serves every hooked layer of the worker process."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.buf = open_shared(create=False)  # the sidecar creates it; a missing buffer means no sidecar
        self.host, dptr = register(self.buf, device)
        self.words_host = self.host[:TABLE_OFF].view(torch.int64)
        self.table_host = self.host[TABLE_OFF:TABLE_OFF + MAX_LAYERS * 4].view(torch.int32)
        shared = device_view(dptr, device)
        self.words = shared[:64].view(torch.int64)
        self.header = shared[SLOT:SLOT + HEADER * 8].view(torch.int64)
        self.px = shared[X_OFF:IDS_OFF].view(torch.bfloat16)
        self.pids = shared[IDS_OFF:W_OFF].view(torch.int32)
        self.pw = shared[W_OFF:OUT_OFF].view(torch.float32)
        self.pout = shared[OUT_OFF:OUT_OFF + MAX_ROWS * HIDDEN * 2].view(torch.bfloat16)
        self.seq = torch.zeros(1, dtype=torch.int64, device=device)
        self.ok = torch.zeros(1, dtype=torch.int32, device=device)
        self.max_rows = MAX_ROWS
        self.timeout = int(os.environ.get("E3_PEER_SPINS", "20000000"))
        # The sidecar may have served an earlier GB300 process: continue from its completed sequence.
        self.seq.fill_(int(self.words_host[1]))
        self.words_host[0] = int(self.words_host[1])

    def serving(self) -> tuple[bool, int, int]:
        return int(self.words_host[7]) == MAGIC, int(self.words_host[8]), int(self.words_host[10])

    def warm_count(self, layer: int) -> int:
        return int(self.table_host[layer])

    def counters(self) -> dict:
        w = self.words_host
        return {"published": int(w[0]), "completed": int(w[1]), "timeouts": int(w[2]), "rows": int(w[3]),
                "tokens": int(w[4]), "wait_ns": int(w[5]), "waits": int(w[6])}

    def send(self, x: torch.Tensor, warm_ids: torch.Tensor, weights: torch.Tensor, layer: int):
        """Pack tokens with a warm route and publish; returns state for finish()."""
        tokens, topk = warm_ids.shape
        if tokens > MAX_ROWS or topk != TOPK:
            raise ValueError(f"E3 peer tier handles up to {MAX_ROWS} tokens of top-{TOPK} routes")
        if x.stride(-1) != 1:
            x = x.contiguous()
        has = (warm_ids >= 0).any(dim=1)
        pos = torch.cumsum(has, 0, dtype=torch.int32) - 1
        rows = has.sum(dtype=torch.int32).reshape(1)
        has = has.to(torch.int8)
        _pack[(tokens,)](
            x, x.stride(0), warm_ids.contiguous(), weights.float().contiguous(), has, pos,
            self.px, self.pids, self.pw, H=HIDDEN, K=TOPK, BLOCK=1024, BLOCK_K=8)
        _publish[(1,)](self.words, self.header, self.seq, layer, rows, tokens, self.pids, K=TOPK, PAD=SMALL_ROWS)
        return has, pos, rows

    def finish(self, output: torch.Tensor, has, pos, rows) -> None:
        _wait[(1,)](self.words, self.seq, rows, self.ok, TIMEOUT=self.timeout)
        _scatter_add[(output.shape[0], triton.cdiv(HIDDEN, 1024))](
            output, output.stride(0), self.pout, has, pos, self.ok, H=HIDDEN, BLOCK=1024)

    def dense_rows(self, tokens: int, has, pos) -> torch.Tensor:
        """FP32 [tokens, HIDDEN] view of the sidecar's last answer (eager-only; for the cosine gate)."""
        out = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=self.device)
        idx = has.nonzero().flatten()
        if idx.numel():
            out[idx] = self.pout[: MAX_ROWS * HIDDEN].view(MAX_ROWS, HIDDEN)[pos[idx].long()].float()
        return out
