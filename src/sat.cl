#ifdef __CDT_PARSER__
#include "OpenCLKernel.hpp"
#define WIDTH 32
#define HEIGHT 32
#define N_COLUMNS 1
#define N_ROWS 1
#define LAST_M 1
#define LAST_N 1
#define BORDER 1
#define CARRY_WIDTH 32
#define CARRY_HEIGHT 1
#define INV_WIDTH 1.0F
#define INPUT_STRIDE 160
#define INV_HEIGHT 1.0F
#include <gpudefs.h>
#else
#include "gpudefs.h"
#endif

//== IMPLEMENTATION ===========================================================

//-- SAT Stage 1 ----------------------------------------------------

//group size: WARP_SIZE X SCHEDULE_OPTIMIZED_N_WARPS
//yBar: rowGroupCount x CARRY_WIDTH
//vHat: colGroupCount x CARRY_HEIGHT
/**
 *CARRY_WIDTH/WARP_SIZE blocks wide
 *
 *Groups/Block size: WARP_SIZE X SCHEDULE_OPTIMIZED_N_WARPS
 *@param input - CARRY_WIDTH x CARRY_HEIGHT
 *@param rowGroupCount - rowGroupCount X CARRY_WIDTH
 *@param vHat - colGroupCount x CARRY_HEIGHT
 */
__kernel
void computeBlockAggregates(const __global float* input, __global float* yBar,
		__global float* vHat, __global float* debugBuf) {

	const size_t yWorkItem = get_local_id(1), xWorkItem = get_local_id(0),
			yGroup = get_group_id(1), xGroup = get_group_id(0), col =
					get_global_id(0), row = get_global_id(1), row0 = row
					- yWorkItem;

	//local memory to store intermediate results, size WARP_SIZE x WARP_SIZE+1
	__local float dataBlock[WARP_SIZE][ WARP_SIZE + 1];

	//pointer to an array of floats of WARP_SIZE + 1, starting at the coordinate of the work item within
	//the sBlock
	__local float (*dataRow)[WARP_SIZE + 1] =
			(__local float (*)[WARP_SIZE + 1]) &dataBlock[yWorkItem][xWorkItem];

	//position the input pointer to this work-item's cell
	input += row * CARRY_WIDTH + col;
	yBar += yGroup * CARRY_WIDTH + col;	//top->bottom output block
	vHat += xGroup * CARRY_WIDTH + row0 + xWorkItem;	//left->right output block

#pragma unroll
	//fill the local data block that's shared between work items
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i++) {
		//copy WARP_SIZE+1 values over from the input
		**dataRow = *input;
		dataRow += SCHEDULE_OPTIMIZED_N_WARPS;
		input += INPUT_STRIDE;
	}
	//if we're in the last few rows of work items, finish up the remaining copying
	if (yWorkItem < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		**dataRow = *input;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (yWorkItem == 0) {

		{   // calculate ybar (aggregate block/group vertically) -----------

			__local float(*dataRow)[WARP_SIZE + 1] =
					(__local float (*)[WARP_SIZE + 1]) &dataBlock[0][xWorkItem];

			float prev = **dataRow;
			++dataRow;

#pragma unroll
			for (int i = 1; i < WARP_SIZE; ++i, ++dataRow)
				**dataRow = prev = **dataRow + prev;

			*yBar = prev;
		}

		{   // calculate vhat (aggregate block/group horizontally) ----------
			__local float *dataRow = dataBlock[xWorkItem];

			float prev = *dataRow;
			++dataRow;

#pragma unroll
			for (int i = 1; i < WARP_SIZE; ++i, ++dataRow)
				prev = *dataRow + prev;

			*vHat = prev;
		}

	}
}

//-- Algorithm SAT Stage 2 ----------------------------------------------------
/**
 * Aggregates the horizontal block-row-wise block column sums into column/block sums along the whole image
 * @param[in] yBar [rowGroupCount x CARRY_WIDTH] - sums of columns in each block row
 * @param[out] ySum [colGroupCount x rowGroupCount] - sums of rows and columns for each block
 */
__kernel
void verticalAggregate(__global float *yBar, __global float *ySum) {

	const size_t yWorkItem = get_local_id(1), xWorkItem = get_local_id(0), xGroup = get_group_id(
			0), col0 = xGroup * MAX_WARPS + yWorkItem, col = col0 * WARP_SIZE + xWorkItem;

	if (col >= CARRY_WIDTH)
		return;

	yBar += col;
	float y = *yBar;
	int ln = HALF_WARP_SIZE + xWorkItem;

	if (xWorkItem == WARP_SIZE - 1)
		ySum += col0;

	volatile __local float dataBlock[ MAX_WARPS][ HALF_WARP_SIZE + WARP_SIZE + 1];

	if (xWorkItem < HALF_WARP_SIZE)
		dataBlock[yWorkItem][xWorkItem] = 0.f;
	else
		dataBlock[yWorkItem][ln] = 0.f;

	for (int n = 1; n < N_ROWS; ++n) {

		// calculate ysum -----------------------
		//TODO: can't we just do inner loop unroll here? I mean, how do we know to stop at ln - 16?
		dataBlock[yWorkItem][ln] = y;

		dataBlock[yWorkItem][ln] += dataBlock[yWorkItem][ln - 1];
		dataBlock[yWorkItem][ln] += dataBlock[yWorkItem][ln - 2];
		dataBlock[yWorkItem][ln] += dataBlock[yWorkItem][ln - 4];
		dataBlock[yWorkItem][ln] += dataBlock[yWorkItem][ln - 8];
		dataBlock[yWorkItem][ln] += dataBlock[yWorkItem][ln - 16];

		if (xWorkItem == WARP_SIZE - 1) {
			*ySum = dataBlock[yWorkItem][ln];
			ySum += N_COLUMNS;
		}

		//TODO:??? fix ybar -> y ??? (left over from original code)-------------------------

		yBar += CARRY_WIDTH;
		y = *yBar += y;

	}

}

//-- Algorithm SAT Stage 3 ----------------------------------------------------
/**
 * Aggregates the horizontal block-row-wise block column sums into column/block sums along the whole image
 * @param[in] yBar [rowGroupCount x CARRY_WIDTH] - sums of columns in each block row
 * @param[out] ySum [colGroupCount x rowGroupCount] - sums of rows and columns for each block
 */
__kernel
void horizontalAggregate(const __global float *ySum, __global float *vHat) {

	const size_t xWorkItem = get_local_id(0), yWorkItem = get_local_id(1), yGroup = get_group_id(
			1), row0 = yGroup * MAX_WARPS + yWorkItem, row = row0 * WARP_SIZE + xWorkItem;

	if (row >= CARRY_HEIGHT)
		return;

	vHat += row;
	float y = 0.f, v = 0.f;

	if (row0 > 0)
		ySum += (row0 - 1) * N_COLUMNS;

	for (int m = 0; m < N_COLUMNS; ++m) {

		// fix vhat -> v -------------------------

		if (row0 > 0) {
			y = *ySum;
			ySum += 1;
		}

		v = *vHat += v + y;
		vHat += CARRY_HEIGHT;

	}

}

//-- Algorithm SAT Stage 4 ----------------------------------------------------

__kernel
void redistributeSAT( __global float *matrix, __global const float *yBar,
		__global const float *vHat) {

	const size_t tx = get_local_id(0), ty = get_local_id(1), bx = get_group_id(
			0), by = get_group_id(1), col = bx * WARP_SIZE + tx, row0 = by
			* WARP_SIZE;

	__local float s_block[ WARP_SIZE][ WARP_SIZE + 1];

	__local float (*bdata)[WARP_SIZE + 1] =
			(__local float (*)[WARP_SIZE + 1]) &s_block[ty][tx];

	matrix += (row0 + ty) * CARRY_WIDTH + col;
	if (by > 0)
		yBar += (by - 1) * CARRY_WIDTH + col;
	if (bx > 0)
		vHat += (bx - 1) * CARRY_HEIGHT + row0 + tx;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *matrix;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		matrix += SCHEDULE_OPTIMIZED_N_WARPS * CARRY_WIDTH;
	}
	if (ty < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *matrix;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ty == 0) {

		{   // calculate y -----------------------
			__local float(*bdata)[WARP_SIZE + 1] = (__local float (*)[WARP_SIZE
					+ 1]) &s_block[0][tx];

			float prev;
			if (by > 0)
				prev = *yBar;
			else
				prev = 0.f;

#pragma unroll
			for (int i = 0; i < WARP_SIZE; ++i, ++bdata)
				**bdata = prev = **bdata + prev;
		}

		{   // calculate x -----------------------
			__local float *bdata = s_block[tx];

			float prev;
			if (bx > 0)
				prev = *vHat;
			else
				prev = 0.f;

#pragma unroll
			for (int i = 0; i < WARP_SIZE; ++i, ++bdata)
				*bdata = prev = *bdata + prev;
		}

	}

	barrier(CLK_LOCAL_MEM_FENCE);

	bdata = (__local float (*)[WARP_SIZE + 1]) &s_block[ty][tx];

	matrix -= (WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS))
			* CARRY_WIDTH;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		*matrix = **bdata;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		matrix += SCHEDULE_OPTIMIZED_N_WARPS * CARRY_WIDTH;
	}
	if (ty < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		*matrix = **bdata;
	}

}

//-- Algorithm SAT Stage 4 (not-in-place) -------------------------------------

__kernel
void algSAT_stage4_not_inplace( __global float *g_out,
		__global const float *g_in, __global const float *g_y,
		__global const float *g_v) {

	const size_t tx = get_local_id(0), ty = get_local_id(1), bx = get_group_id(
			0), by = get_group_id(1), col = bx * WARP_SIZE + tx, row0 = by
			* WARP_SIZE;
	__local float s_block[ WARP_SIZE][ WARP_SIZE + 1];

	__local float (*bdata)[WARP_SIZE + 1] =
			(__local float (*)[WARP_SIZE + 1]) &s_block[ty][tx];

	g_in += (row0 + ty) * CARRY_WIDTH + col;
	if (by > 0)
		g_y += (by - 1) * CARRY_WIDTH + col;
	if (bx > 0)
		g_v += (bx - 1) * CARRY_HEIGHT + row0 + tx;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *g_in;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		g_in += SCHEDULE_OPTIMIZED_N_WARPS * CARRY_WIDTH;
	}
	if (ty < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *g_in;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ty == 0) {

		{   // calculate y -----------------------
			__local float(*bdata)[WARP_SIZE + 1] = (__local float (*)[WARP_SIZE
					+ 1]) &s_block[0][tx];

			float prev;
			if (by > 0)
				prev = *g_y;
			else
				prev = 0.f;

#pragma unroll
			for (int i = 0; i < WARP_SIZE; ++i, ++bdata)
				**bdata = prev = **bdata + prev;
		}

		{   // calculate x -----------------------
			__local float *bdata = s_block[tx];

			float prev;
			if (bx > 0)
				prev = *g_v;
			else
				prev = 0.f;

#pragma unroll
			for (int i = 0; i < WARP_SIZE; ++i, ++bdata)
				*bdata = prev = *bdata + prev;
		}

	}

	barrier(CLK_LOCAL_MEM_FENCE);

	bdata = (__local float (*)[WARP_SIZE + 1]) &s_block[ty][tx];

	g_out += (row0 + ty) * CARRY_WIDTH + col;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		*g_out = **bdata;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		g_out += SCHEDULE_OPTIMIZED_N_WARPS * CARRY_WIDTH;
	}
	if (ty < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		*g_out = **bdata;
	}

}
