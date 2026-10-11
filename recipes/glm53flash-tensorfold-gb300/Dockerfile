# TensorFold v0.6.0 (c4646171) + Mia's GLM-5.3-Flash recipe patches (b5808ac, 107) + ours (9001-9003), for one GB300.
# Mirrors the recipe's inline Dockerfile (scripts/prepare.sh L180-190): TensorFold pinned by commit, our patches applied
# last, the bench client + public prompt pool baked in. Network at BUILD time only; run.sh serves with --network none.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
# GitHub tarball of the pinned commit (same tree as git+...@c4646171; works in images without git)
ARG TF_SPEC=https://github.com/ashhart/TensorFold/archive/c4646171139ee8a3c38103eaa1699dad226ec12b.tar.gz
ARG EXTRAS="av==18.1.0 xgrammar>=0.2.8,<0.3"
RUN (command -v patch >/dev/null || (apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/*)) && \
    pip install --no-cache-dir --upgrade "${TF_SPEC}" && pip install --no-cache-dir ${EXTRAS} && tensorfold --version
COPY patches/mia /opt/tf-patches/mia
ARG OURS=ours
COPY patches/${OURS} /opt/tf-patches/ours
RUN cd "$(python3 -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/mia/*.patch /opt/tf-patches/ours/*.patch; do echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python3 -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine, tensorfold.vision.glm, av, xgrammar; \
from tensorfold.families.glm5_next import CUDA_TP; from tensorfold.families.glm5_next.cuda.tp import SIZES; \
assert 1 in CUDA_TP and 1 in SIZES, (CUDA_TP, SIZES); print('tp1 ok', CUDA_TP, SIZES)"
COPY mia-scripts /opt/mia-scripts
COPY b1 /opt/b1
ENV HF_HOME=/root/.cache/huggingface TORCH_EXTENSIONS_DIR=/out/cache/torch_extensions TRITON_CACHE_DIR=/out/cache/triton \
    HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1
WORKDIR /workspace
