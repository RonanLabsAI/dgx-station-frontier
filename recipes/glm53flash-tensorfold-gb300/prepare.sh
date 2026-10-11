#!/usr/bin/env bash
# Prepare the GLM-5.3-Flash / TensorFold GB300 recipe: pinned inputs, sha256-verified, then the image.
#   1. Mia's GLM-5.3-Flash recipe patches (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ b5808ac)
#   2. checkpoint Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold @ 078455ff (175.7 GB) and drafter
#      incoai/GLM-5.3-Flash-DFlash2 @ bf582e4e (CC BY-NC-ND 4.0: non-commercial; downloaded, never redistributed)
#   3. public prompt pool (CNN/DailyMail 3.0.0 test, ShareGPT V3, GSM8K test) at pinned revisions
#   4. the image: BASE (ngc: nvcr.io/nvidia/pytorch:26.07-py3 | vllm: a CUDA 13.0 image, for drivers that cannot use
#      CUDA 13.3 forward compatibility) + TensorFold v0.6.0 (c4646171) + Mia's 107 patches + ours (9001-9003)
# Needs: docker with the NVIDIA runtime, git, curl, python3 with huggingface_hub, pandas, tokenizers. Network: this script
# only; run.sh serves with --network none.
# usage: MODELS_DIR=/path/with/~200GB BASE=ngc ./prepare.sh
set -euo pipefail
cd "$(dirname "$0")"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"; BASE="${BASE:-ngc}"
MIA_SHA=b5808ac620c795ec9c0640f3ca2a2f7be21aaa63
CKPT_ID=Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold; CKPT_REV=078455ffe6472f9a52fbc1139f58b9db2881b25c
DFL_ID=incoai/GLM-5.3-Flash-DFlash2; DFL_REV=bf582e4eacc1810f76656d1811693ff6c6737d2a
case "$BASE" in
  ngc)  BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3 ;;
  vllm) BASE_IMAGE="${VLLM_BASE_IMAGE:?set VLLM_BASE_IMAGE to a CUDA 13.0 PyTorch image with python3 and nvcc}" ;;
  *) echo "BASE=ngc|vllm" >&2; exit 2 ;;
esac
TAG="tensorfold-glm53flash-gb300:v0.6.0-c4646171-mia-b5808ac-9001-9003-$BASE"
CTX=ctx; rm -rf "$CTX"; mkdir -p "$CTX/patches/ours" "$CTX/b1/pool" "$CTX/mia-scripts" "$MODELS_DIR"

echo "== 1. Mia's recipe @ $MIA_SHA"
git clone -q https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold.git "$CTX/mia"
git -C "$CTX/mia" checkout -q "$MIA_SHA"
mv "$CTX/mia/patches" "$CTX/patches/mia"; cp "$CTX/mia/scripts/config.sh" "$CTX/mia-scripts/"; rm -rf "$CTX/mia"
[[ $(ls "$CTX/patches/mia" | wc -l) == 107 ]] || { echo "expected 107 patches" >&2; exit 1; }
cp patches/900[123]-*.patch "$CTX/patches/ours/"

echo "== 2. weights (pinned, then every LFS file sha256-checked against the Hub)"
fetch() {   # fetch REPO REV DIR
  python3 - "$1" "$2" "$3" <<'EOF'
import hashlib, json, sys, urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download
repo, rev, out = sys.argv[1:4]
snapshot_download(repo, revision=rev, local_dir=out, max_workers=16)
meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"))
bad = 0
for s in meta["siblings"]:
    want = (s.get("lfs") or {}).get("sha256")
    if not want:
        continue
    h = hashlib.sha256()
    with open(Path(out) / s["rfilename"], "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != want:
        print("MISMATCH", s["rfilename"]); bad += 1
if bad:
    sys.exit(1)
(Path(out) / ".revision").write_text(rev + "\n"); (Path(out) / ".verified").write_text("ok\n")
print("verified", repo, rev)
EOF
}
fetch "$CKPT_ID" "$CKPT_REV" "$MODELS_DIR/Mia-AiLab__GLM-5.3-Flash-EXL3-4bpw-TensorFold"
fetch "$DFL_ID" "$DFL_REV" "$MODELS_DIR/incoai__GLM-5.3-Flash-DFlash2-bf582e4e"

echo "== 3. public prompt pool"
SRC=pool-src; mkdir -p "$SRC"          # outside the docker build context
curl -sL -o "$SRC/test.parquet" "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/96df5e686bee6baa90b8bee7c28b81fa3fa6223d/3.0.0/test-00000-of-00001.parquet"
curl -sL -o "$SRC/sg.json" "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"
cp "$MODELS_DIR/Mia-AiLab__GLM-5.3-Flash-EXL3-4bpw-TensorFold/tokenizer.json" "$SRC/glm-tokenizer.json"
curl -sL -o "$CTX/b1/gsm8k_test.jsonl" "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
( cd "$SRC" && sha256sum -c --quiet <<EOF
04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e  test.parquet
35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4  sg.json
19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d  glm-tokenizer.json
EOF
)
python3 bench/build_pool_glm.py "$SRC/test.parquet" "$SRC/sg.json" "$SRC/glm-tokenizer.json" "$CTX/b1/pool" > /dev/null
python3 - <<'EOF'
import json
a = json.load(open("bench/pool-manifest.json"))["files"]; b = json.load(open("ctx/b1/pool/manifest.json"))["files"]
assert a == b, "prompt pool differs from the published manifest"
print("prompt pool matches bench/pool-manifest.json")
EOF
cp bench/bench_tf.py bench/gsm8k_tf.py bench/smoke_tf.py "$CTX/b1/"

echo "== 4. image $TAG"
cp Dockerfile "$CTX/"
docker build -t "$TAG" --build-arg BASE_IMAGE="$BASE_IMAGE" "$CTX"
echo "ready: MODELS_DIR=$MODELS_DIR IMAGE_TAG=$TAG ./run.sh"
