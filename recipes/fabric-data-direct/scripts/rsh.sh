#!/bin/bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# OpenMPI rsh agent (runs inside the head dd-bench container): remote host -> ssh $RSH_USER@host docker exec dd-bench bash -c CMD
h=$1; shift
exec ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i /root/.ssh/id_ed25519 \
  "${RSH_USER:?}"@"$h" "docker exec -i dd-bench bash -c $(printf %q "$*")"
