// Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
// minimal STREAM (copy/scale/add/triad), OpenMP, doubles. usage: stream <N elements> <reps>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc,char**argv){
  long n=atol(argv[1]); int reps=atoi(argv[2]);
  double *a=malloc(n*8),*b=malloc(n*8),*c=malloc(n*8); double s=3.0;
  #pragma omp parallel for
  for(long i=0;i<n;i++){a[i]=1;b[i]=2;c[i]=0;}
  double best[4]={1e9,1e9,1e9,1e9}; const char* nm[4]={"Copy","Scale","Add","Triad"}; double by[4]={16,16,24,24};
  for(int r=0;r<reps;r++){
    double t=omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<n;i++) c[i]=a[i];
    t=omp_get_wtime()-t; if(t<best[0])best[0]=t;
    t=omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<n;i++) b[i]=s*c[i];
    t=omp_get_wtime()-t; if(t<best[1])best[1]=t;
    t=omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<n;i++) c[i]=a[i]+b[i];
    t=omp_get_wtime()-t; if(t<best[2])best[2]=t;
    t=omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<n;i++) a[i]=b[i]+s*c[i];
    t=omp_get_wtime()-t; if(t<best[3])best[3]=t;
  }
  printf("threads=%d n=%ld (%.1f GiB/array)\n",omp_get_max_threads(),n,n*8.0/(1<<30));
  for(int k=0;k<4;k++) printf("RESULT stream_%s_GBps %.1f\n",nm[k],by[k]*n/best[k]/1e9);
  return a[7]>0?0:1;
}
