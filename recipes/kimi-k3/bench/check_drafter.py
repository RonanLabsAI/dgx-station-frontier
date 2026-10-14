#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""check_drafter.py DRAFTER_DIR [--target-config PATH] [--no-size] -- CPU-only check of the K3 DSpark drafter.

Reads ONLY config.json and the safetensors header (8-byte length + JSON; ~7.5 KB), never the weights, and no torch:
  * every tensor the K3DSparkModel / K3DSparkForCausalLM layout needs (vLLM af7f9488 models/kimi_k3/nvidia/
    dspark_mla.py, the checkpoint's HF names before WeightsMapper) exists with the shape config.json implies, and
    nothing unexpected is present;
  * every dtype equals config torch_dtype (BF16);
  * total parameters, and data bytes == file size - 8 - header length (no truncated download);
  * K3DSparkConfig's own refusals (q_lora_rank set, draft_vocab_size == vocab_size, num_target_layers ==
    len(target_layer_ids), no mla_use_nope / output gate / qk norm);
  * with --target-config (the served K3 config.json, or a {"text_config": {...}} subset): hidden size, vocab and
    layer count agree with the target, every target_layer_id is a real target layer, and target MLA latent geometry
    matches (kv_lora_rank, qk_rope_head_dim).
Prints one JSON object; exit 0 if every check passes, 1 otherwise.
"""
import json
import os
import struct
import sys

DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4, "U8": 1}
TORCH_TO_ST = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def expected_shapes(c):
    h, inter, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    nh, ql, kvl = c["num_attention_heads"], c["q_lora_rank"], c["kv_lora_rank"]
    nope, rope, vd = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    vocab, dvocab = c["vocab_size"], c.get("draft_vocab_size") or c["vocab_size"]
    e = {
        "context_proj.weight": [h, c["target_hidden_size"] * c["num_target_layers"]],
        "context_norm.weight": [h],
        "final_norm.weight": [h],
        "embed_tokens.weight": [vocab, h],   # in the checkpoint; dropped by the vLLM mapper (target embedding shared)
        "markov_head.markov_w1.weight": [dvocab, c["markov_rank"]],
        "markov_head.markov_w2.weight": [dvocab, c["markov_rank"]],
    }
    if c.get("enable_confidence_head"):
        e["confidence_head.proj.weight"] = [1, h + (c["markov_rank"] if c.get("confidence_head_with_markov") else 0)]
        e["confidence_head.proj.bias"] = [1]
    for i in range(L):
        p = f"layers.{i}."
        e.update({
            p + "input_layernorm.weight": [h],
            p + "post_attention_layernorm.weight": [h],
            p + "mlp.gate_proj.weight": [inter, h],
            p + "mlp.up_proj.weight": [inter, h],
            p + "mlp.down_proj.weight": [h, inter],
            p + "self_attn.q_a_proj.weight": [ql, h],
            p + "self_attn.q_a_layernorm.weight": [ql],
            p + "self_attn.q_b_proj.weight": [nh * (nope + rope), ql],
            p + "self_attn.kv_a_proj_with_mqa.weight": [kvl + rope, h],
            p + "self_attn.kv_a_layernorm.weight": [kvl],
            p + "self_attn.kv_b_proj.weight": [nh * (nope + vd), kvl],
            p + "self_attn.o_proj.weight": [h, nh * vd],
        })
    return e


def config_refusals(c):
    """K3DSparkConfig.__init__ (vllm/transformers_utils/configs/k3_dspark.py at af7f9488) raises on these."""
    bad = [n for n in ("mla_use_nope", "mla_use_output_gate", "mla_use_qk_norm", "dspark_bonus_anchor") if c.get(n)]
    if c.get("q_lora_rank") is None:
        bad.append("q_lora_rank=None")
    if (c.get("draft_vocab_size") or c["vocab_size"]) != c["vocab_size"]:
        bad.append("draft_vocab_size != vocab_size")
    tl = c.get("target_layer_ids")
    if not tl or c.get("num_target_layers") != len(tl):
        bad.append("target_layer_ids / num_target_layers mismatch")
    return bad


def check(drafter_dir, target_config=None, size_check=True):
    c = json.load(open(os.path.join(drafter_dir, "config.json")))
    st = os.path.join(drafter_dir, "model.safetensors")
    n, hdr = read_header(st)
    meta = hdr.pop("__metadata__", None)
    want_dt = TORCH_TO_ST.get(c.get("torch_dtype", "bfloat16"), "BF16")
    exp = expected_shapes(c)
    missing = sorted(k for k in exp if k not in hdr)
    unexpected = sorted(k for k in hdr if k not in exp)
    wrong_shape = {k: {"have": hdr[k]["shape"], "want": exp[k]} for k in exp if k in hdr and hdr[k]["shape"] != exp[k]}
    dtypes = sorted({v["dtype"] for v in hdr.values()})
    params = sum(_prod(v["shape"]) for v in hdr.values())
    data_bytes = sum(_prod(v["shape"]) * DT_BYTES.get(v["dtype"], 0) for v in hdr.values())
    span = max((v["data_offsets"][1] for v in hdr.values()), default=0)
    file_size = os.path.getsize(st)
    embed = _prod(hdr["embed_tokens.weight"]["shape"]) if "embed_tokens.weight" in hdr else 0
    out = {
        "drafter": drafter_dir, "architectures": c.get("architectures"), "model_type": c.get("model_type"),
        "header_bytes": n, "tensors": len(hdr), "metadata": meta, "dtypes": dtypes, "want_dtype": want_dt,
        "params_total": params, "params_loaded_by_vllm": params - embed, "params_embed_shared_with_target": embed,
        "data_bytes": data_bytes, "data_span": span, "file_size": file_size,
        "missing": missing, "unexpected": unexpected, "wrong_shape": wrong_shape,
        "config_refusals": config_refusals(c),
        "layers": c["num_hidden_layers"], "target_layer_ids": c.get("target_layer_ids"),
    }
    checks = {
        "all_tensors_present": not missing,
        "no_unexpected_tensors": not unexpected,
        "all_shapes_match_config": not wrong_shape,
        "all_dtypes_match_config": dtypes == [want_dt],
        "data_span_equals_tensor_bytes": span == data_bytes,
        "config_accepted_by_K3DSparkConfig": not out["config_refusals"],
    }
    if size_check:
        checks["file_size_equals_8_plus_header_plus_data"] = file_size == 8 + n + data_bytes
    if target_config:
        t = json.load(open(target_config))
        t = t.get("text_config", t)
        out["target"] = {k: t.get(k) for k in ("hidden_size", "num_hidden_layers", "vocab_size", "kv_lora_rank",
                                               "qk_rope_head_dim", "num_nextn_predict_layers")}
        checks.update({
            "target_hidden_matches": c["target_hidden_size"] == t["hidden_size"] == c["hidden_size"],
            "target_vocab_matches": c["vocab_size"] == t["vocab_size"],
            "target_layers_match": c.get("target_num_hidden_layers") == t["num_hidden_layers"],
            "target_layer_ids_in_range": all(0 <= i < t["num_hidden_layers"] for i in c["target_layer_ids"]),
            "mla_latent_matches_target": (t.get("kv_lora_rank") in (None, c["kv_lora_rank"])
                                          and t.get("qk_rope_head_dim") in (None, c["qk_rope_head_dim"])),
        })
    out["checks"] = checks
    out["ok"] = all(checks.values())
    return out


def _prod(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    tc = None
    if "--target-config" in argv:
        tc = argv[argv.index("--target-config") + 1]
        args = [a for a in args if a != tc]
    out = check(args[0], tc, size_check="--no-size" not in argv)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
