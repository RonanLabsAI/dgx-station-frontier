# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""CPU tests for the three-tier placement builder (python3 -m pytest tests/test_rowmap.py)."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_rowmap_e3 as b  # noqa: E402


def test_offload_matches_e1_log():
    # E1 logged "Total CPU offloaded parameters: 200.81 GiB" with --cpu-offload-gb 200 on both ranks.
    layers = b.offloaded_layers(200, 2)
    assert layers == list(range(34))
    w13, w2, _, _ = b.expert_bytes()
    assert abs(34 * 192 * (w13 + w2) / b.GIB - 200.8125) < 1e-6


def test_expert_bytes_match_checkpoint():
    # layers.N.ffn.experts.E.{w1,w3}.weight I8 [3072, 3584], w2 [7168, 1536]; scales [3072, 224] / [7168, 96]
    w13, w2, s13, s2 = b.expert_bytes()
    assert w13 == 2 * 3072 * 3584 and w2 == 7168 * 1536 and s13 == 2 * 3072 * 224 and s2 == 7168 * 96


def test_rowmap_partition(tmp_path):
    out = tmp_path / "rm.json"
    subprocess.check_call([sys.executable, os.path.join(ROOT, "tools", "build_rowmap_e3.py"), "--counts",
                           os.path.join(ROOT, "calibration", "routes-e1-gsm8k100.json"), "--capacity", "500",
                           "--out", str(out)], stdout=subprocess.DEVNULL)
    rm = json.load(open(out))
    assert sorted(int(k) for k in rm["layers"]) == list(range(34))
    for rank in (0, 1):
        ids = [e for v in rm["layers"].values() for e in v["warm"] if rank * 192 <= e < rank * 192 + 192]
        assert len(ids) == 500
    for v in rm["layers"].values():
        assert len(set(v["warm"])) == len(v["warm"]) and all(0 <= e < 384 for e in v["warm"])
        assert sum(e < 192 for e in v["warm"]) >= 8 and sum(e >= 192 for e in v["warm"]) >= 8
