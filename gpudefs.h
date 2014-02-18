#ifndef GPUDEFS_H
#define GPUDEFS_H

#define WARP_SIZE 32 // Warp size (defines b x b block size where b = WS)

#define HALF_WARP_SIZE 16 // Half Warp Size
#define DEFAULT_N_WARPS 8 // Default number of warps (computational block height)
#define CARRY_HEAVY_N_WARPS 7 // Carry-heavy number of warps (computational block height for some kernels)
#define OPTIMIZED_N_WARPS 6 // Optimized number of warps (computational block height for some kernels)
#define DEFAULT_N_BLOCKS 6 // Default number of blocks per SM (minimum blocks per SM launch bounds)
#define OPTIMIZED_N_BLOCKS 5 // Optimized number of blocks per SM (minimum blocks per SM for some kernels)
#define MAX_N_THREADS 192 // Maximum number of threads per block with 8 blocks per SM
#define MAX_BLOCK_OPTIMIZE 8 // Maximum number of blocks per SM using optimize or maximum warps
#define CARRY_HEAVY_N_BLOCKS 7 // Carry-heavy number of blocks per SM using default number of warps
#define MAX_WARPS 6 // Maximum number of warps per block with 8 blocks per SM (with all warps computing)
#define SCHEDULE_OPTIMIZED_N_WARPS 5 // Dual-scheduler optimized number of warps per block (with 8 blocks per SM and to use the dual scheduler with 1 computing warp)
#define MAX_BLOCKS_HALF 3 // Maximum number of blocks per SM using half-warp size

#endif // GPUDEFS_H
