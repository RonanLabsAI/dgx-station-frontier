# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
import torch, time, sys
T=float(sys.argv[1]); a=torch.randn(8192,8192,dtype=torch.bfloat16,device="cuda"); b=torch.randn_like(a)
t=time.time(); n=0
while time.time()-t<T:
    for _ in range(50): c=a@b
    torch.cuda.synchronize(); n+=50
dt=time.time()-t; print(f"RESULT gemm_bf16_8192_TFLOPS {2*8192**3*n/dt/1e12:.0f}", flush=True)
