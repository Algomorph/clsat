#ifdef __CDT_PARSER__
#include "OpenCLKernel.hpp"
#define WIDTH 1
#define HEIGHT 1
#define M_SIZE 1
#define N_SIZE 1
#define LAST_M 1
#define LAST_N 1
#define BORDER 1
#define CARRY_WIDTH 1
#define CARRY_HEIGHT 1
#define INV_WIDTH 1.0F
#define INV_HEIGHT 1.0F
#include <gpudefs.h>
#else
#include "gpudefs.h"
#endif

//== CONSTANT DECLARATION
__constant int c_width = WIDTH, c_height = HEIGHT, c_m_size = M_SIZE, c_n_size =
N_SIZE, c_last_m = LAST_M, c_last_n = LAST_N, c_border = BORDER, c_carry_width =
		CARRY_WIDTH, c_carry_height = CARRY_HEIGHT;

__constant float c_inv_width = INV_WIDTH, c_inv_height = INV_HEIGHT;

//== IMPLEMENTATION ===========================================================

//-- Algorithm SAT Stage 1 ----------------------------------------------------

__kernel
void algSAT_stage1(__global float* g_in, __global float* g_ybar,
		__global float* g_vhat) {

	const size_t work_item_y = get_local_id(1), work_item_x = get_local_id(0),
			group_y = get_group_id(1), group_x = get_group_id(0), col = group_x
					* WARP_SIZE + work_item_x, row0 = group_y * WARP_SIZE;

	__local float s_block[ WARP_SIZE][ WARP_SIZE + 1];
	float (*bdata)[WARP_SIZE + 1] =
			(float (*)[WARP_SIZE + 1]) &s_block[work_item_y][work_item_x];

	g_in += (row0 + work_item_y) * c_width + col;
	g_ybar += group_y * c_width + col;
	g_vhat += group_x * c_height + row0 + work_item_x;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *g_in;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		g_in += SCHEDULE_OPTIMIZED_N_WARPS * c_width;
	}
	if (work_item_y < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *g_in;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (work_item_y == 0) {

		{   // calculate ybar -----------------------
			float(*bdata)[WARP_SIZE + 1] =
					(float (*)[WARP_SIZE + 1]) &s_block[0][work_item_x];

			float prev = **bdata;
			++bdata;

#pragma unroll
			for (int i = 1; i < WARP_SIZE; ++i, ++bdata)
				**bdata = prev = **bdata + prev;

			*g_ybar = prev;
		}

		{   // calculate vhat -----------------------
			__local float *bdata = s_block[work_item_x];

			float prev = *bdata;
			++bdata;

#pragma unroll
			for (int i = 1; i < WARP_SIZE; ++i, ++bdata)
				prev = *bdata + prev;

			*g_vhat = prev;
		}

	}
}

//-- Algorithm SAT Stage 2 ----------------------------------------------------

__kernel
void algSAT_stage2(__global float *g_ybar, __global float *g_ysum) {

	const size_t ty = get_local_id(1), tx = get_local_id(0), bx = get_group_id(
			0), col0 = bx * MAX_WARPS + ty, col = col0 * WARP_SIZE + tx;

	if (col >= c_width)
		return;

	g_ybar += col;
	float y = *g_ybar;
	int ln = HALF_WARP_SIZE + tx;

	if (tx == WARP_SIZE - 1)
		g_ysum += col0;

	volatile __local float s_block[ MAX_WARPS][ HALF_WARP_SIZE + WARP_SIZE + 1];

	if (tx < HALF_WARP_SIZE)
		s_block[ty][tx] = 0.f;
	else
		s_block[ty][ln] = 0.f;

	for (int n = 1; n < c_n_size; ++n) {

		// calculate ysum -----------------------

		s_block[ty][ln] = y;

		s_block[ty][ln] += s_block[ty][ln - 1];
		s_block[ty][ln] += s_block[ty][ln - 2];
		s_block[ty][ln] += s_block[ty][ln - 4];
		s_block[ty][ln] += s_block[ty][ln - 8];
		s_block[ty][ln] += s_block[ty][ln - 16];

		if (tx == WARP_SIZE - 1) {
			*g_ysum = s_block[ty][ln];
			g_ysum += c_m_size;
		}

		// fix ybar -> y -------------------------

		g_ybar += c_width;
		y = *g_ybar += y;

	}

}

//-- Algorithm SAT Stage 3 ----------------------------------------------------

__kernel
void algSAT_stage3( __global const float *g_ysum, __global float *g_vhat) {

	const size_t tx = get_local_id(0), ty = get_local_id(1), by = get_group_id(
			1), row0 = by * MAX_WARPS + ty, row = row0 * WARP_SIZE + tx;

	if (row >= c_height)
		return;

	g_vhat += row;
	float y = 0.f, v = 0.f;

	if (row0 > 0)
		g_ysum += (row0 - 1) * c_m_size;

	for (int m = 0; m < c_m_size; ++m) {

		// fix vhat -> v -------------------------

		if (row0 > 0) {
			y = *g_ysum;
			g_ysum += 1;
		}

		v = *g_vhat += v + y;
		g_vhat += c_height;

	}

}

//-- Algorithm SAT Stage 4 ----------------------------------------------------

__kernel
void algSAT_stage4_inplace( __global float *g_inout, __global const float *g_y,
		__global const float *g_v) {

	const size_t tx = get_local_id(0), ty = get_local_id(1), bx = get_group_id(
			0), by = get_group_id(1), col = bx * WARP_SIZE + tx, row0 = by
			* WARP_SIZE;

	__local float s_block[ WARP_SIZE][ WARP_SIZE + 1];

	float (*bdata)[WARP_SIZE + 1] = (float (*)[WARP_SIZE + 1]) &s_block[ty][tx];

	g_inout += (row0 + ty) * c_width + col;
	if (by > 0)
		g_y += (by - 1) * c_width + col;
	if (bx > 0)
		g_v += (bx - 1) * c_height + row0 + tx;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *g_inout;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		g_inout += SCHEDULE_OPTIMIZED_N_WARPS * c_width;
	}
	if (ty < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *g_inout;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ty == 0) {

		{   // calculate y -----------------------
			float(*bdata)[WARP_SIZE + 1] =
					(float (*)[WARP_SIZE + 1]) &s_block[0][tx];

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

	bdata = (float (*)[WARP_SIZE + 1]) &s_block[ty][tx];

	g_inout -= (WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS)) * c_width;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		*g_inout = **bdata;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		g_inout += SCHEDULE_OPTIMIZED_N_WARPS * c_width;
	}
	if (ty < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		*g_inout = **bdata;
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

	float (*bdata)[WARP_SIZE + 1] = (float (*)[WARP_SIZE + 1]) &s_block[ty][tx];

	g_in += (row0 + ty) * c_width + col;
	if (by > 0)
		g_y += (by - 1) * c_width + col;
	if (bx > 0)
		g_v += (bx - 1) * c_height + row0 + tx;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *g_in;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		g_in += SCHEDULE_OPTIMIZED_N_WARPS * c_width;
	}
	if (ty < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		**bdata = *g_in;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ty == 0) {

		{   // calculate y -----------------------
			float(*bdata)[WARP_SIZE + 1] =
					(float (*)[WARP_SIZE + 1]) &s_block[0][tx];

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

	bdata = (float (*)[WARP_SIZE + 1]) &s_block[ty][tx];

	g_out += (row0 + ty) * c_width + col;

#pragma unroll
	for (int i = 0; i < WARP_SIZE - (WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS);
			i += SCHEDULE_OPTIMIZED_N_WARPS) {
		*g_out = **bdata;
		bdata += SCHEDULE_OPTIMIZED_N_WARPS;
		g_out += SCHEDULE_OPTIMIZED_N_WARPS * c_width;
	}
	if (ty < WARP_SIZE % SCHEDULE_OPTIMIZED_N_WARPS) {
		*g_out = **bdata;
	}

}
