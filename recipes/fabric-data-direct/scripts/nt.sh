#!/bin/bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# nt.sh <SET> <OUTDIR> <test> [nccl-tests args...]  -- run an nccl-test across both Stations from the rank-0 dd-bench container.
# env RANK0_IP RANK1_IP (rail IPs), RAIL_NET (e.g. 10.10.1.0/24), IFN, HCA, RSH_USER (ssh user on the peer).
SET=$1; O=$2; T=$3; shift 3
NL=/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib
B="-x LD_LIBRARY_PATH=$NL -x NCCL_SOCKET_IFNAME=${IFN:-enP1p3s0f1np1} -x NCCL_IB_HCA==${HCA:-mlx5_1}:1 -x NCCL_IB_DISABLE=0 -x NCCL_NET_GDR_LEVEL=SYS -x NCCL_DMABUF_ENABLE=1 -x NCCL_IB_TC=106 -x NCCL_DEBUG=${DBG:-WARN}"
[ -n "$SUBSYS" ] && B="$B -x NCCL_DEBUG_SUBSYS=$SUBSYS"
case $SET in
  E0|E2) X="";;
  E1) X="-x NCCL_ALGO=Ring -x NCCL_PROTO=Simple -x NCCL_MIN_NCHANNELS=8 -x NCCL_MAX_NCHANNELS=8 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_SPLIT_DATA_ON_QPS=1";;
  E3) X="-x NCCL_PROTO=LL -x NCCL_ALGO=Ring -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1";;
  E4) X="-x NCCL_PROTO=LL -x NCCL_ALGO=Ring -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2";;
  E5) X="-x NCCL_PROTO=LL128 -x NCCL_ALGO=Ring -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2";;
  E6) X="-x NCCL_PROTO=LL,LL128,Simple -x NCCL_MIN_NCHANNELS=8 -x NCCL_MAX_NCHANNELS=8 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_SPLIT_DATA_ON_QPS=1";;
  E7) X="-x NCCL_TUNER_PLUGIN=/dd/libnccl-tuner-example.so -x NCCL_TUNER_CONFIG_FILE=/dd/tuner.csv -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_SPLIT_DATA_ON_QPS=1";;
  E10) X="-x NCCL_PROTO=LL128 -x NCCL_ALGO=Ring -x NCCL_MIN_NCHANNELS=4 -x NCCL_MAX_NCHANNELS=4";;
  E11) X="-x NCCL_TUNER_PLUGIN=/dd/libnccl-tuner-example.so -x NCCL_TUNER_CONFIG_FILE=/dd/tuner2.csv -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_SPLIT_DATA_ON_QPS=1";;
  E8) X="-x NCCL_PROTO=Simple -x NCCL_ALGO=Ring -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1";;
  *) X="$EXTRA";;
esac
mkdir -p $O; L=$O/${SET}_${T}_${TAG:-x}.log
echo "# $(date '+%F %T %Z') SET=$SET $X args=$*" > $L
docker exec -e RSH_USER=${RSH_USER:-$USER} dd-bench mpirun --allow-run-as-root -np 2 --host ${RANK0_IP:?}:1,${RANK1_IP:?}:1 --bind-to none \
  --mca plm_rsh_agent /dd/rsh.sh --mca oob_tcp_if_include ${RAIL_NET:?} --mca pml ob1 --mca btl self,tcp \
  --mca btl_tcp_if_include $RAIL_NET $B $X /dd/src/nccl-tests/build/$T "$@" >> $L 2>&1
echo "exit=$? -> $L"; grep -E "^\s+[0-9]+ " $L | awk '{print $1, $6, $8, $10, $13}' | tail -40
