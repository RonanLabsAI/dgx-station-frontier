#!/bin/bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# ctr.sh up <OVL 0|1> | down   -- start/stop the dd-bench container (SGLang dev image, CUDA 13.0, NCCL 2.30.7 pip) on THIS Station.
# OVL=1 bind-mounts the host MOFED libibverbs/libmlx5 (~/hostrdma, Data Direct API) OVER the image's real files + provider dir.
set -e
# env: IMG (default: the SGLang dev image digest we measured with), DDDIR (scripts + tuner + nccl-tests, mounted at /dd),
#      UVERBS (the rail's uverbs device), HOSTRDMA (host MOFED libs with the Data Direct symbol, see README).
NAME=dd-bench; IMG=${IMG:-lmsysorg/sglang@sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d}
DDDIR=${DDDIR:-$HOME/dd}; UVERBS=${UVERBS:-uverbs1}; HOSTRDMA=${HOSTRDMA:-$HOME/hostrdma}
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
case "$1" in
 down) docker rm -f $NAME >/dev/null 2>&1 || true; echo "down $(hostname)";;
 up)
  OV=(); if [ "$2" = 1 ]; then
    nm -D $HOSTRDMA/libmlx5.so.1 | grep -q mlx5dv_get_data_direct_sysfs_path || { echo "hostrdma not staged"; exit 3; }
    # the image's real file names (rdma-core 50 in this image); resolve them with readlink for other images
    OV=(-v $HOSTRDMA/libibverbs.so.1:/usr/lib/aarch64-linux-gnu/libibverbs.so.1.14.50.0:ro
        -v $HOSTRDMA/libmlx5.so.1:/usr/lib/aarch64-linux-gnu/libmlx5.so.1.24.50.0:ro
        -v /usr/lib/aarch64-linux-gnu/libibverbs:/usr/lib/aarch64-linux-gnu/libibverbs:ro
        -v /etc/libibverbs.d:/etc/libibverbs.d:ro); fi
  SSHM=(); [ -f ~/.ssh/id_ed25519 ] && SSHM=(-v $HOME/.ssh:/hostssh:ro)
  docker rm -f $NAME >/dev/null 2>&1 || true
  docker run -d --name $NAME --network host --ipc host --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
    --gpus device=$GB300 --device /dev/infiniband/$UVERBS $([ -c /dev/infiniband/rdma_cm ] && echo --device /dev/infiniband/rdma_cm) "${OV[@]}" "${SSHM[@]}" \
    -v $DDDIR:/dd -v $HOME/bench:/bench --entrypoint sleep "$IMG" infinity >/dev/null
  docker exec $NAME bash -c 'if [ -d /hostssh ]; then mkdir -p /root/.ssh && cp /hostssh/id_ed25519 /root/.ssh/ && chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_ed25519; fi'
  echo "up $(hostname) ovl=$2 gpu=$GB300";;
 *) echo "usage: ctr.sh up 0|1 | down"; exit 1;;
esac
