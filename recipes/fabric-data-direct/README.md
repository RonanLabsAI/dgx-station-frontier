# Fabric: Data Direct in containers, a per-size NCCL tuner, and the small-message floor

Two DGX Stations, one 400G DAC between their ConnectX-8 ports (RoCEv2, no switch). This recipe is what we add to every
two-Station launcher, and what we measured about the link.

**The short version:**

1. **Data Direct is a bind-mount, not a reboot.** The engine images ship an rdma-core without the Data Direct API, so
   NCCL logs `dlvsym failed` and falls back to a slower host path. Mounting the host's MOFED `libibverbs` and
   `libmlx5` over the image's files fixes it.
2. **Add a per-size tuner.** Data Direct alone helps bandwidth but NCCL then picks a poor protocol for mid-size
   messages and loses decode throughput. `tuner3.csv` fixes that.
3. **Use traffic class 106.** On this link only priority 3 is lossless (PFC); `NCCL_IB_TC=106` puts RoCE there.
   Without it RoCE goes out on priority 0, which has no PFC.
4. **Small messages are latency-bound, and NCCL settings cannot move it.** A graph-captured all-reduce of a decode-sized
   message costs about 31-33 µs on the engine path in every setting we tried, 11.3-13.1 µs above nccl-tests. A second
   400G rail would add bandwidth, not lower this floor; single-user decode across Stations needs a different collective.

Credit: catid published the Data Direct container overlay first ([dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks)).
That repository has no licence, so nothing is copied from it; `scripts/dd-overlay.sh` and `scripts/ctr.sh` are ours.

## Results

<!-- results -->
| End to end: GLM-5.3 NVFP4, SGLang TP2 + EP2 across both Stations | Stock (no Data Direct) | Data Direct + tuner3 + `TC 106` | Change, % | Receipt |
|---|---|---|---|---|
| Prefill, one `~68K`-token prompt, tok/s | 8,033 | 11,173 | +39.1 | [log](../../logs/fabric-data-direct/fused-glm-dd-ab.log) |
| Decode C1, tok/s | 79.3 | 80.7 | +1.8 | [log](../../logs/fabric-data-direct/fused-glm-dd-ab.log) |
| Decode C8 aggregate, tok/s | 410.4 | 414.4 | +1.0 | [log](../../logs/fabric-data-direct/fused-glm-dd-ab.log) |
| Decode C16 aggregate, tok/s | 600.3 | 634.5 | +5.7 | [log](../../logs/fabric-data-direct/fused-glm-dd-ab.log) |
| Decode C32 aggregate, tok/s | 1,000.2 | 1,068.7 | +6.8 | [log](../../logs/fabric-data-direct/fused-glm-dd-ab.log) |
<!-- /results -->

Same session, a fresh boot per arm. With the earlier `tuner2.csv` (LL only up to `64 KB`) C8 dropped to 397.2 tok/s
(−3.2%); `tuner3.csv` moves the LL/LL128 boundary to `128 KB` and removes that regression. In an earlier session,
**Data Direct without any tuner cost 8.1% at C32** (993.7 to 912.9 tok/s, [log](../../logs/fabric-data-direct/fused-glm-dd-only.log)).
The decode harness is J-M-Recipes' `flash_bench.py` (MIT; not vendored): a short prose prompt with a nonce, 256 output
tokens, three repetitions. Prefill uses `scripts/prefill_bench.py` with unique-prefix prompts, so nothing is cached.

<!-- results -->
| Link measurement | Value | Receipt |
|---|---|---|
| nccl-tests all_reduce `2 GiB` bus bandwidth, GB/s: no Data Direct / Data Direct / + Ring, Simple, `8` channels, `4` QPs / + tuner2 | 26.62 / 42.35 / 48.73 / 47.03 | [log](../../logs/fabric-data-direct/nccl-tests-2GiB.log) |
| torch all_reduce `128 MB`, Gb/s: no Data Direct / Data Direct / + Ring, Simple, `8` channels | 189 / 321 / 351 | [log](../../logs/fabric-data-direct/torch-probe-128MB.log) |
| GPU-memory RDMA write with Data Direct, one way, Gb/s | 392.06 | [log](../../logs/fabric-data-direct/rdma-verbs.log) |
| `8-byte` RDMA write latency, µs: host memory / GPU memory with Data Direct | 1.6 / 2.21 | [log](../../logs/fabric-data-direct/rdma-verbs.log) |
| Engine-path all-reduce, `14 KB`, CUDA graph, µs: torch / SGLang pynccl / torch + tuner2 | 31.67 / 32.88 / 31.13 | [log](../../logs/fabric-data-direct/lambda-engine-path.log) |
| Same, with Ring, Simple, `8` channels, `4` QPs (a published two-Station setting), µs | 47.62 | [log](../../logs/fabric-data-direct/lambda-engine-path.log) |
| Engine-path all-reduce, `192 KB` (C16-sized), µs: NCCL auto / tuner2 | 55.68 / 47.98 | [log](../../logs/fabric-data-direct/lambda-engine-path.log) |
| nccl-tests all_reduce, `16 KB`, µs: auto / tuner2 | 20.84 / 19.22 | [log](../../logs/fabric-data-direct/lambda-engine-path.log) |
| nccl-tests all_reduce, `256 KB`, µs: auto / LL128 forced, `4` channels | 50.5 / 33.52 | [log](../../logs/fabric-data-direct/lambda-engine-path.log) |
| New RDMA sequence errors over the whole TC on/off matrix | 0 | [log](../../logs/fabric-data-direct/seq-errors-tc106.log) |
<!-- /results -->

What these say:

- **Data Direct is a bandwidth fix.** It lifts large collectives and prefill; it does not change small-message latency.
- **The Ring/Simple/8-channel setting** (good for `2 GiB` bandwidth) **adds about 16 µs to every small collective**,
  which hurts every decode step. Do not use it for decode.
- **LL128 is the fastest protocol from `128 KB` up on this link, and NCCL never chooses it on its own.** A per-size
  tuner is the only way to get it.
- **The 11-13 µs gap** between nccl-tests and the engine path (graph-captured, in-place, framework-allocated buffers) is
  unexplained. Every NCCL knob we tried lands at 31-33 µs on the engine path.
- **Sequence errors:** an earlier session (without TC 106) logged 12,053 `packet_seq_err` on one side and the same number
  of `out_of_sequence` on the other. A full matrix of probes with and without TC 106, including an 8-channel, 4-QP
  stress run, added none. TC 106 stays the default because it is the only lossless class.

## Fabric note: prefill chunk size, the one-rail bound, clean decode absolutes (2026-10-09)

**Chunked prefill.** We swept SGLang's `--chunked-prefill-size` on the fused GLM-5.3 two-Station server (rail 1 only,
Data Direct + tuner3 + `TC 106`, a fresh boot per point, `--max-prefill-tokens 65536`). The server was our fused
one-shot build: the stock image above plus two local patches for the decode-path small all-reduce, not in this
release. Large prefill all-reduces still go through NCCL, and the `8192` point reproduces the stock arm's 11,173
tok/s within a tenth of a percent.

<!-- results -->
| `--chunked-prefill-size` | `64K` prefill, C1, tok/s (median of `3`) | `64K x 4` concurrent, tok/s | `8K` prefill, C1, tok/s | `Rail 1` transmit while busy / peak, GB/s | Receipt |
|---|---|---|---|---|---|
| `8192` (the setting in use) | 11,169 | 11,175 | 9,337 | 21.7 / 23.7 | [log](../../logs/fabric-data-direct/fused-glm-clean-rerun.log) |
| `16384` | 11,420 | 11,436 | 11,668 | 22.4 / 25.1 | [log](../../logs/fabric-data-direct/fused-glm-clean-rerun.log) |
| **`32768`** | **11,854** (+6.1%) | 11,843 | 11,687 (+25.2%) | 23.1 / 25.5 | [log](../../logs/fabric-data-direct/fused-glm-clean-rerun.log) |
| `65536` | out of memory on the first `64K` request (`6.56 GiB` allocation at memory fraction `0.93`) | | | | [log](../../logs/fabric-data-direct/fused-glm-clean-rerun.log) |
<!-- /results -->

- **`32768` is the best setting:** +6.1% on `64K`-token prefill, and C1 decode unchanged (+0.1%, inside run-to-run
  noise; [log](../../logs/fabric-data-direct/fused-glm-clean-rerun.log)). Our launchers still use `8192`.
- **Prefill is bound by NCCL throughput on one rail, not by the line.** Rail 1 is busy for the whole prefill at about
  23.1 GB/s ([log](../../logs/fabric-data-direct/fused-glm-clean-rerun.log)), 0.46x of the `400G` line rate (`50 GB/s`), and the chunk size barely moves it. About `157`
  all-reduces of `64K x 6144 x 2 B` per prefill come to roughly `126 GB`; at that rate that is about the measured
  prefill time. **A second rail is the main lever**; more NCCL channels or queue pairs for the large all-reduces is the
  other (nccl-tests reaches 48.73 GB/s bus bandwidth at `2 GiB` on this one rail, [log](../../logs/fabric-data-direct/nccl-tests-2GiB.log)).

**Clean decode absolutes (stock fused launcher).** The same server as the stock column's Data Direct + tuner3 arm,
benched on catid's shape instead of `flash_bench`: random ids, 8,192 in / 1,024 out, `2 x C` requests, a fresh seed per
run, decode = C x 1000 / mean TPOT. Every prefill batch of the session logged zero cached tokens. These are not
comparable with the `flash_bench` rows above (short prose prompt, 256 output tokens).

<!-- results -->
| Fused GLM-5.3, stock launcher, catid shape | C1 | C8 | C16 | C32 | Receipt |
|---|---|---|---|---|---|
| Decode tok/s (aggregate) | 84.8 | 422.6 | 588.0 | 977.1 | [log](../../logs/fabric-data-direct/fused-glm-clean-rerun.log) |
<!-- /results -->

## Hardware and software

| Item | Value |
|---|---|
| Link | one 400G DAC, ConnectX-8 port 1 on each Station (`mlx5_1`), RoCEv2, GID 3, firmware 40.47.1088, no switch |
| QoS (as shipped) | trust DSCP; PFC on priority 3 only; DSCP 24-31 map to priority 3. TC 106 = DSCP 26 |
| Image | `lmsysorg/sglang@sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d` (CUDA 13.0.88, torch 2.13.0+cu130); NCCL 2.30.7 is what loads at run time (`torch.cuda.nccl.version()` reports the compile-time 2.29.7) |
| Tools | nccl-tests @`afd59ab` built in the image with MPI against NCCL 2.30.7 for `sm_103`; perftest 25.10 on the hosts |
| Tuner plugin | NCCL's example tuner plugin, built from the NCCL source tree @`12df1a11` (`libnccl-tuner-example.so`, tuner API v5/v6) |

## How to use it

1. **Stage the host libraries once per Station.** Copy (dereferencing links) the host MOFED `libibverbs.so.1` and
   `libmlx5.so.1` into `~/hostrdma/` and confirm the Data Direct symbol is there:
   ```bash
   mkdir -p ~/hostrdma && cp -L /usr/lib/aarch64-linux-gnu/{libibverbs.so.1,libmlx5.so.1} ~/hostrdma/
   nm -D ~/hostrdma/libmlx5.so.1 | grep mlx5dv_get_data_direct_sysfs_path
   ```
2. **Build the tuner plugin** from the NCCL source (the example tuner under the NCCL tree's tuner-plugin examples) and
   put `libnccl-tuner-example.so` plus [`tuner/tuner3.csv`](tuner/tuner3.csv) in `~/nccl-tuner/`.
   CSV columns: collective, min bytes, max bytes, algorithm, protocol, channels, nodes, ranks, pipeline ops, registered
   buffers (`-1` = any).
3. **Source the overlay in your launcher** and add the arguments to `docker run`:
   ```bash
   IMAGE=<engine image> source scripts/dd-overlay.sh        # fills DDARGS; DD=0 makes it a no-op
   docker run ... --device /dev/infiniband/uverbs1 "${DDARGS[@]}" ... "$IMAGE" ...
   ```
   Keep your rail settings too: `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA==mlx5_1:1`, `NCCL_NET_GDR_LEVEL=SYS`,
   `NCCL_DMABUF_ENABLE=1`. [`../dsv4-pro-two-stations/launch/launch-dsv4pro.sh`](../dsv4-pro-two-stations/launch/launch-dsv4pro.sh)
   is a complete example.
4. **Check every rank's log** (`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET`) for
   `NET/IB: Data Direct DMA Interface is detected for device mlx5_1`, `Successfully loaded external tuner plugin`, and
   no `dlvsym failed` lines.

## Reproducing the link measurements

`scripts/ctr.sh up 1` starts a bench container (the SGLang image above, with or without the overlay) on each Station;
`scripts/nt.sh <SET> <outdir> all_reduce_perf -b 8 -e 2G -f 2 -g 1` runs nccl-tests across both through OpenMPI
(`rsh.sh` reaches the peer container over ssh; `ctr.sh` mounts `~/.ssh` read-only for that, so use a dedicated key);
`scripts/tp.sh lam_probe.py <log> [NCCL_X=...]` and `tp.sh lam_pynccl.py` measure the engine-path latency;
`tp.sh probe_dd.py` is the `128 MB` probe. The `SET` names (E0 to E11) are the environment sets in the receipts.
Set `RANK0_IP`, `RANK1_IP`, `RAIL_NET`, `PEER_SSH` and `RSH_USER` for your two hosts.

Not included: the fused GLM-5.3 two-Station launcher itself. Ours was adapted from an unlicensed public script and
must be rewritten before it can be published; the Data Direct block it used is `scripts/dd-overlay.sh`.
