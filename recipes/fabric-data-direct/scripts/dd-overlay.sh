#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# dd-overlay.sh -- source this from a docker-run launcher to get Data Direct + per-size NCCL tuner + lossless traffic class
# for ANY image (SGLang or vLLM). It fills the bash array DDARGS with docker-run arguments:
#   * bind-mounts the host's MOFED libibverbs/libmlx5 (which export mlx5dv_get_data_direct_sysfs_path) OVER the image's
#     real library files, plus the provider dir and /etc/libibverbs.d. Image file names are resolved with readlink, so the
#     same block works across images;
#   * mounts the NCCL example tuner plugin + a tuner CSV and sets NCCL_TUNER_PLUGIN / NCCL_TUNER_CONFIG_FILE;
#   * sets NCCL_IB_QPS_PER_CONNECTION=4, NCCL_IB_SPLIT_DATA_ON_QPS=1, NCCL_IB_TC=106 (DSCP 26 -> priority 3, the only
#     PFC-protected class on our link; without a TC, RoCE goes out on priority 0, which is lossy).
# Usage:   IMAGE=<image> source dd-overlay.sh ; docker run ... "${DDARGS[@]}" ... "$IMAGE" ...
# Env:     DD=1 (0 = no-op, reproduces the stock launch), HOSTRDMA (default ~/hostrdma), TUNERDIR (default ~/nccl-tuner,
#          must hold libnccl-tuner-example.so and the CSV), TUNER_CSV (default tuner3.csv)
# Check:   every rank's NCCL INFO log must show "NET/IB: Data Direct DMA Interface is detected for device <hca>" and no
#          "dlvsym failed" lines (needs NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,NET).
DD=${DD:-1}; DDARGS=(); TUNER_CSV=${TUNER_CSV:-tuner3.csv}
HOSTRDMA=${HOSTRDMA:-$HOME/hostrdma}; TUNERDIR=${TUNERDIR:-$HOME/nccl-tuner}
if [ "$DD" = 1 ]; then
  : "${IMAGE:?set IMAGE before sourcing dd-overlay.sh}"
  nm -D "$HOSTRDMA/libmlx5.so.1" | grep -q mlx5dv_get_data_direct_sysfs_path \
    || { echo "$HOSTRDMA not staged: cp -L the host's libibverbs.so.1 and libmlx5.so.1 there (DD=0 to skip)"; return 3 2>/dev/null || exit 3; }
  for f in libnccl-tuner-example.so "$TUNER_CSV"; do
    [ -s "$TUNERDIR/$f" ] || { echo "$TUNERDIR/$f missing"; return 3 2>/dev/null || exit 3; }
  done
  _C=$(docker create "$IMAGE" true); _T=$(mktemp -d)
  docker cp "$_C:/usr/lib/aarch64-linux-gnu/libibverbs.so.1" "$_T/" >/dev/null
  docker cp "$_C:/usr/lib/aarch64-linux-gnu/libmlx5.so.1" "$_T/" >/dev/null
  docker rm "$_C" >/dev/null; _IBV=$(readlink "$_T/libibverbs.so.1"); _MLX=$(readlink "$_T/libmlx5.so.1"); rm -rf "$_T"
  [ -n "$_IBV" ] && [ -n "$_MLX" ] || { echo "could not resolve the image's rdma library names"; return 3 2>/dev/null || exit 3; }
  [ -c /dev/infiniband/rdma_cm ] && DDARGS+=(--device /dev/infiniband/rdma_cm)
  DDARGS+=(-v "$HOSTRDMA/libibverbs.so.1:/usr/lib/aarch64-linux-gnu/$_IBV:ro"
           -v "$HOSTRDMA/libmlx5.so.1:/usr/lib/aarch64-linux-gnu/$_MLX:ro"
           -v /usr/lib/aarch64-linux-gnu/libibverbs:/usr/lib/aarch64-linux-gnu/libibverbs:ro
           -v /etc/libibverbs.d:/etc/libibverbs.d:ro
           -v "$TUNERDIR:/opt/nccl-tuner:ro"
           -e NCCL_TUNER_PLUGIN=/opt/nccl-tuner/libnccl-tuner-example.so
           -e NCCL_TUNER_CONFIG_FILE=/opt/nccl-tuner/$TUNER_CSV
           -e NCCL_IB_QPS_PER_CONNECTION=4 -e NCCL_IB_SPLIT_DATA_ON_QPS=1 -e NCCL_IB_TC=106)
fi
