#include "cuda_runtime.h"
#include "device_launch_paramters.h"
#include <stdio.h>
#include <omp.h>

#define TILE_WIDTH 16

// a simple version of matrix multiplication which issues redundant loads from off-chip
// global memory