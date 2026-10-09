# Microbenchmark atlas: one DGX Station GB300

Rooflines for placing model weights on a DGX Station: how fast the GB300 reads its own HBM, Grace memory over
NVLink-C2C, and how fast the RTX PRO 6000 reaches Grace over PCIe. Measured on one Station, 2026-10-06 PT, with the
other GPU idle unless stated. All sources are in [`src/`](src/).

## Results

<!-- results -->
| Path | Measurement | GB/s | Receipt |
|---|---|---|---|
| Grace to GB300 over C2C | `cudaMemcpy`, pinned host memory, host to device / device to host | 353.6 / 375.1 | [log](../../logs/atlas-microbench/gb300-membench.log) |
| Grace to GB300 over C2C | GPU kernel reading pinned Grace memory directly (zero-copy) | 352.8 | [log](../../logs/atlas-microbench/gb300-membench.log) |
| Grace to GB300, managed memory | GPU reading managed memory preferred-on-CPU / prefetch host to device | 91.0 / 78.4 | [log](../../logs/atlas-microbench/gb300-membench.log) |
| GB300 HBM | kernel read | 6,481.2 | [log](../../logs/atlas-microbench/gb300-membench.log) |
| Grace memory | STREAM triad, `48` threads | 305.4 | [log](../../logs/atlas-microbench/grace-stream.log) |
| Grace to RTX PRO 6000 over PCIe | pinned, host to device / device to host | 55.1 / 57.2 | [log](../../logs/atlas-microbench/rtx-membench.log) |
| Grace to RTX PRO 6000 over PCIe | pageable, host to device / device to host | 16.9 / 14.9 | [log](../../logs/atlas-microbench/rtx-membench.log) |
<!-- /results -->

<!-- results -->
| Compute and power | Value | Receipt |
|---|---|---|
| One MoE expert (SwiGLU, hidden `7168`, intermediate `2048`, bf16): GPU reading the weights from Grace over C2C vs the `72`-core Grace CPU computing it, speed-up at batch `1` / `16` / `64` | 7.58x / 6.91x / 7.12x | [log](../../logs/atlas-microbench/expertbench.log) |
| GB300 dense bf16 GEMM (`8192` cubed), RTX PRO 6000 idle / serving, TFLOPS | 1,030 / 1,033 | [log](../../logs/atlas-microbench/gemm-power.log) |
| GB300 peak power during that GEMM / `power.limit`, W | 1,081.46 / 1,300 | [log](../../logs/atlas-microbench/gemm-power.log) |
<!-- /results -->

What to take from it:

- **Pin your host memory.** Pinned C2C copies run at 353.6 GB/s; the same data as managed memory reads at about a
  quarter of that (91.0 GB/s). This is the most common way to lose a Grace tier.
- **Zero-copy is as fast as a copy.** A GPU kernel reading pinned Grace memory in place gets 352.8 GB/s, so experts
  can be computed straight from Grace without staging them into HBM first.
- **Do not compute experts on the Grace CPU.** At batch 1 to 64 the GPU, reading the same weights over C2C, is
  6.9-7.6x faster. This is a negative result for designs that run cold experts on the host CPU.
- **The RTX PRO 6000 is a PCIe device:** 55-57 GB/s pinned to Grace, about a sixth of C2C, and pageable transfers lose
  two thirds of that. Keep what it serves resident in its own memory (as the expert sidecars in the recipes do).
- **The two GPUs do not share a power cap in practice.** The GB300 reached 1,081.46 W under dense GEMM whether the RTX
  PRO 6000 was idle or busy, and delivered the same TFLOPS.

## Reproduce

```bash
nvcc -O3 -arch=sm_103a -o membench src/membench.cu          # use sm_120 for the RTX PRO 6000
./membench 8192 10                                          # GB300: 8 GiB buffers, 10 reps
CUDA_VISIBLE_DEVICES=<rtx> ./membench 256 20                # RTX PRO 6000: 256 MiB, 20 reps
gcc -O3 -fopenmp -o stream src/stream.c && OMP_NUM_THREADS=48 ./stream 400000000 10
python3 src/expertbench.py                                  # needs torch with CUDA (we ran it in a vLLM image)
python3 src/gemmload.py 30 & URL=... MODEL=... src/rtxload.sh   # GEMM with the RTX PRO 6000 busy; sample power with nvidia-smi
```

Our builds used host CUDA 13.2 (`membench`) and gcc (`stream`); `expertbench.py` and `gemmload.py` ran in
`vllm/vllm-openai@sha256:f29125bcc6d276ed38a67c9dfd4e51ccbaba09ad92847a0913c0df38d9c05c71` (torch 2.13.0+cu130).
