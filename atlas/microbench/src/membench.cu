// Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
// membench.cu - C2C / HBM / PCIe microbench. usage: membench <size_MiB> <reps> [skip_managed]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s: %s (line %d)\n",#x,cudaGetErrorString(e),__LINE__);}}while(0)

__global__ void readk(const int4* __restrict__ p, size_t n, int* out){
  size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  int acc=0;
  for(; i<n; i+=st){ int4 v=p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if(acc==0x7fffffff) out[0]=acc;
}
__global__ void writek(int4* p, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=st){ p[i]=make_int4(i,1,2,3); }
}
static int SMS=0;
static float time_read(const void* p, size_t bytes, int reps, int* out){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  size_t n=bytes/16; int blocks=SMS*8;
  readk<<<blocks,512>>>((const int4*)p,n,out); CK(cudaDeviceSynchronize());
  cudaEventRecord(a); for(int r=0;r<reps;r++) readk<<<blocks,512>>>((const int4*)p,n,out); cudaEventRecord(b);
  CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms,a,b); return (float)(bytes*(double)reps/(ms/1e3)/1e9);
}
static float time_write(void* p, size_t bytes, int reps){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  size_t n=bytes/16; int blocks=SMS*8;
  writek<<<blocks,512>>>((int4*)p,n); CK(cudaDeviceSynchronize());
  cudaEventRecord(a); for(int r=0;r<reps;r++) writek<<<blocks,512>>>((int4*)p,n); cudaEventRecord(b);
  CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms,a,b); return (float)(bytes*(double)reps/(ms/1e3)/1e9);
}
static float time_copy(void* d, const void* s, size_t bytes, int reps, cudaMemcpyKind k){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  CK(cudaMemcpy(d,s,bytes,k));
  cudaEventRecord(a); for(int r=0;r<reps;r++) CK(cudaMemcpyAsync(d,s,bytes,k)); cudaEventRecord(b);
  CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms,a,b); return (float)(bytes*(double)reps/(ms/1e3)/1e9);
}
int main(int argc,char**argv){
  size_t mib=atoll(argv[1]); int reps=atoi(argv[2]); int skipm=argc>3?atoi(argv[3]):0;
  size_t bytes=mib<<20;
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0)); SMS=pr.multiProcessorCount;
  int pma=0, ats=0, cma=0, hrp=0;
  cudaDeviceGetAttribute(&pma,cudaDevAttrPageableMemoryAccess,0);
  cudaDeviceGetAttribute(&ats,cudaDevAttrPageableMemoryAccessUsesHostPageTables,0);
  cudaDeviceGetAttribute(&cma,cudaDevAttrConcurrentManagedAccess,0);
  cudaDeviceGetAttribute(&hrp,cudaDevAttrCanUseHostPointerForRegisteredMem,0);
  printf("device=%s SMs=%d cc=%d.%d size=%zu MiB reps=%d pageableAccess=%d usesHostPT=%d concManaged=%d hostPtrRegistered=%d\n",
    pr.name,SMS,pr.major,pr.minor,mib,reps,pma,ats,cma,hrp);
  int* out; CK(cudaMalloc(&out,64));
  void *d, *h, *pg;
  CK(cudaMalloc(&d,bytes)); CK(cudaHostAlloc(&h,bytes,cudaHostAllocMapped)); pg=malloc(bytes); memset(pg,1,bytes); memset(h,1,bytes);
  printf("RESULT memcpy_H2D_pinned_GBps %.1f\n", time_copy(d,h,bytes,reps,cudaMemcpyHostToDevice));
  printf("RESULT memcpy_D2H_pinned_GBps %.1f\n", time_copy(h,d,bytes,reps,cudaMemcpyDeviceToHost));
  printf("RESULT memcpy_H2D_pageable_GBps %.1f\n", time_copy(d,pg,bytes,reps>3?3:reps,cudaMemcpyHostToDevice));
  printf("RESULT memcpy_D2H_pageable_GBps %.1f\n", time_copy(pg,d,bytes,reps>3?3:reps,cudaMemcpyDeviceToHost));
  void* hd; CK(cudaHostGetDevicePointer(&hd,h,0));
  printf("RESULT zerocopy_kernel_read_pinned_GBps %.1f\n", time_read(hd,bytes,reps,out));
  printf("RESULT zerocopy_kernel_write_pinned_GBps %.1f\n", time_write(hd,bytes,reps));
  if(pma){ printf("RESULT ats_kernel_read_pageable_malloc_GBps %.1f\n", time_read(pg,bytes,reps,out)); }
  printf("RESULT hbm_kernel_read_GBps %.1f\n", time_read(d,bytes,reps,out));
  printf("RESULT hbm_kernel_write_GBps %.1f\n", time_write(d,bytes,reps));
  { void* d2; if(cudaMalloc(&d2,bytes)==cudaSuccess){ float g=time_copy(d2,d,bytes,reps,cudaMemcpyDeviceToDevice); printf("RESULT hbm_D2D_copy_GBps_rw %.1f (read+write %.1f)\n", g, 2*g); cudaFree(d2);} }
  if(!skipm){
    void* m; CK(cudaMallocManaged(&m,bytes)); memset(m,1,bytes);
    // first GPU touch after CPU init: migration or remote access depending on mode
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); readk<<<SMS*8,512>>>((const int4*)m,bytes/16,out); cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms,a,b); printf("RESULT managed_first_gpu_read_after_cpu_init_GBps %.1f\n", bytes/(ms/1e3)/1e9);
    printf("RESULT managed_steady_gpu_read_GBps %.1f\n", time_read(m,bytes,reps,out));
    // force host residency then read remotely (preferred location CPU)
    cudaMemLocation cpu; cpu.type=cudaMemLocationTypeHost; cpu.id=0;
    cudaMemLocation gpu; gpu.type=cudaMemLocationTypeDevice; gpu.id=0;
    CK(cudaMemAdvise(m,bytes,cudaMemAdviseSetPreferredLocation,cpu));
    CK(cudaMemAdvise(m,bytes,cudaMemAdviseSetAccessedBy,gpu));
    CK(cudaMemPrefetchAsync(m,bytes,cpu,0)); CK(cudaDeviceSynchronize());
    printf("RESULT managed_gpu_read_prefCPU_accessedBy_GBps %.1f\n", time_read(m,bytes,reps,out));
    CK(cudaMemAdvise(m,bytes,cudaMemAdviseUnsetPreferredLocation,cpu));
    cudaEventRecord(a); CK(cudaMemPrefetchAsync(m,bytes,gpu,0)); cudaEventRecord(b); CK(cudaEventSynchronize(b));
    cudaEventElapsedTime(&ms,a,b); printf("RESULT managed_prefetch_H2D_GBps %.1f\n", bytes/(ms/1e3)/1e9);
    cudaEventRecord(a); CK(cudaMemPrefetchAsync(m,bytes,cpu,0)); cudaEventRecord(b); CK(cudaEventSynchronize(b));
    cudaEventElapsedTime(&ms,a,b); printf("RESULT managed_prefetch_D2H_GBps %.1f\n", bytes/(ms/1e3)/1e9);
    cudaFree(m);
  }
  return 0;
}
