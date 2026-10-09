#!/bin/bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# tp.sh <script.py> <LOGFILE> [extra NCCL env as K=V ...]  -- run a 2-rank torch script in dd-bench on both Stations (run on rank 0).
# env PEER_SSH (user@rank-1 rail IP), RANK0_IP, IFN, HCA.
S=$1; L=$2; shift 2
IFN=${IFN:-enP1p3s0f1np1}; HCA=${HCA:-mlx5_1}
E=(-e RANK0_IP=${RANK0_IP:?} -e NCCL_SOCKET_IFNAME=$IFN -e GLOO_SOCKET_IFNAME=$IFN -e NCCL_IB_HCA==$HCA:1 -e NCCL_IB_DISABLE=0
   -e NCCL_NET_GDR_LEVEL=SYS -e NCCL_DMABUF_ENABLE=1 -e NCCL_DEBUG=${DBG:-WARN})
[ -n "$SUBSYS" ] && E+=(-e NCCL_DEBUG_SUBSYS=$SUBSYS)
for kv in "$@"; do E+=(-e "$kv"); done
mkdir -p $(dirname $L); echo "# $(date '+%F %T %Z') $S env: $*" > $L
ssh ${PEER_SSH:?} "timeout 300 docker exec -e RANK=1 ${E[*]} dd-bench python3 /dd/$S" > $L.rank1 2>&1 &
P=$!
timeout 300 docker exec -e RANK=0 "${E[@]}" dd-bench python3 /dd/$S >> $L 2>&1; echo "rank0 exit=$?" >> $L
wait $P; echo "rank1 exit=$?" >> $L
grep -E "LAM|busbw|first|exit|Data Direct|dlvsym|Error|error" $L $L.rank1 | head -40
