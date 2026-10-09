#!/bin/bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
export RANK=$OMPI_COMM_WORLD_RANK
exec python3 /dd/lam_pynccl.py
