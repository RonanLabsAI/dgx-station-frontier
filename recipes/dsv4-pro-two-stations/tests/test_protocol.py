# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Protocol tests. CPU part: layout + per-rank selection + handshake (needs torch + triton importable, e.g. inside the
vLLM image). GPU part (E3_GPU_TEST=1): the GB300-side kernels against a NumPy fake sidecar on a shared buffer."""
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hook"))
os.environ.setdefault("E3_SHM", "/dev/shm/e3_test_peer")
import peer_tier_v4 as pt  # noqa: E402


def test_layout():
    assert pt.TABLE_OFF + pt.MAX_LAYERS * 4 <= pt.SLOT
    assert pt.SLOT + pt.HEADER * 8 <= pt.X_OFF
    assert pt.IDS_OFF - pt.X_OFF == pt.MAX_ROWS * pt.HIDDEN * 2
    assert pt.TOTAL_BYTES % 4096 == 0 and pt.TOTAL_BYTES >= pt.OUT_OFF + pt.MAX_ROWS * pt.HIDDEN * 2


def test_rank_warm_and_hash():
    rm = {"layers": {"0": {"warm": [5, 200, 3, 191, 192]}, "1": {"warm": [300]}}}
    r0, r1 = pt.rank_warm(rm, 0, 192), pt.rank_warm(rm, 1, 192)
    assert r0 == {0: [3, 5, 191]} and r1 == {0: [192, 200], 1: [300]}
    assert pt.warm_hash(r0) != pt.warm_hash(r1) and pt.warm_hash(r0) == pt.warm_hash({0: [3, 5, 191]})


@pytest.mark.skipif(os.environ.get("E3_GPU_TEST") != "1", reason="set E3_GPU_TEST=1 on a GPU host")
def test_kernels_against_fake_sidecar():
    import numpy as np
    import torch

    dev = torch.device("cuda", 0)
    buf = pt.open_shared(create=True)
    host, _ = pt.register(buf, dev)
    host[:pt.X_OFF].zero_()
    words = np.frombuffer(buf, dtype=np.int64, count=16)
    header = np.frombuffer(buf, dtype=np.int64, count=pt.HEADER, offset=pt.SLOT)
    words[7] = pt.MAGIC
    x_host = host[pt.X_OFF:pt.IDS_OFF].view(torch.bfloat16)
    ids_host = host[pt.IDS_OFF:pt.W_OFF].view(torch.int32)
    w_host = host[pt.W_OFF:pt.OUT_OFF].view(torch.float32)
    out_host = host[pt.OUT_OFF:pt.OUT_OFF + pt.MAX_ROWS * pt.HIDDEN * 2].view(torch.bfloat16)
    H, K = pt.HIDDEN, pt.TOPK
    stop = False
    seen = []

    def fake():  # out[r] = sum_k w[r,k] * (id+1) * x[r] over live routes
        last = int(words[0])
        while not stop:
            seq = int(words[0])
            if seq == last or int(header[0]) != seq:
                continue
            rows = int(header[2])
            seen.append((int(header[1]), rows))
            if rows:
                x = x_host[: rows * H].view(rows, H).float()
                ids = ids_host[: rows * K].view(rows, K)
                w = w_host[: rows * K].view(rows, K)
                f = (torch.where(ids >= 0, w * (ids + 1).float(), torch.zeros_like(w))).sum(1, keepdim=True)
                out_host[: rows * H].copy_((x * f).to(torch.bfloat16).view(-1))
            words[1] = seq
            last = seq

    th = threading.Thread(target=fake, daemon=True)
    th.start()
    try:
        peer = pt.PeerTierV4(dev)
        T = 37
        x = torch.randn(T, H + 64, device=dev, dtype=torch.bfloat16)[:, :H]   # strided rows, like a padded input
        wid = torch.full((T, K), -1, dtype=torch.int32, device=dev)
        wid[::3, 1] = 4
        wid[5, 2] = 0
        w = torch.rand(T, K, device=dev)
        y = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
        y0 = y.float().clone()
        has, pos, rows = peer.send(x, wid, w, layer=7)
        peer.finish(y, has, pos, rows)
        torch.cuda.synchronize()
        f = torch.where(wid >= 0, w * (wid + 1).float(), torch.zeros_like(w)).sum(1, keepdim=True)
        expect = y0 + (x.float() * f).to(torch.bfloat16).float()
        assert int(peer.ok.item()) == 1 and int(rows.item()) == int((wid >= 0).any(1).sum())
        assert torch.allclose(y.float(), expect, atol=0.05, rtol=0.02)
        assert seen[-1] == (7, int(rows.item())) and peer.counters()["timeouts"] == 0
    finally:
        stop = True
        th.join(timeout=2)
        os.unlink(pt.PATH)
