#ifndef GPUDEFS_H
#define GPUDEFS_H

#define WS 32 // Warp size (defines b x b block size where b = WS)

#define HWS 16 // Half Warp Size
#define DW 8 // Default number of warps (computational block height)
#define CHW 7 // Carry-heavy number of warps (computational block height for some kernels)
#define OW 6 // Optimized number of warps (computational block height for some kernels)
#define DNB 6 // Default number of blocks per SM (minimum blocks per SM launch bounds)
#define ONB 5 // Optimized number of blocks per SM (minimum blocks per SM for some kernels)
#define MTS 192 // Maximum number of threads per block with 8 blocks per SM
#define MBO 8 // Maximum number of blocks per SM using optimize or maximum warps
#define CHB 7 // Carry-heavy number of blocks per SM using default number of warps
#define MW 6 // Maximum number of warps per block with 8 blocks per SM (with all warps computing)
#define SOW 5 // Dual-scheduler optimized number of warps per block (with 8 blocks per SM and to use the dual scheduler with 1 computing warp)
#define MBH 3 // Maximum number of blocks per SM using half-warp size

#endif // GPUDEFS_H
