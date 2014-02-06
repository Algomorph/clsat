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
#define inv_width 1.0F
#define inv_height 1.0F
#endif

#include <gpudefs.h>
namespace sat {

//== CONSTANT DECLARATION
__constant int c_width = WIDTH,
	c_height = HEIGHT,
	c_m_size = M_SIZE,
	c_n_size = N_SIZE,
	c_last_m = LAST_M,
	c_last_n = LAST_N,
	c_border = BORDER,
	c_carry_width = CARRY_WIDTH,
	c_carry_height = CARRY_HEIGHT;

__constant float c_inv_width, c_inv_height,
    c_b0, c_a1, c_a2, c_inv_b0,
    c_AbF, c_AbR, c_HARB_AFP;

//== IMPLEMENTATION ===========================================================

//-- Algorithm SAT Stage 1 ----------------------------------------------------

__kernel
void algSAT_stage1(__global const float* g_in,
		__global float* g_ybar,
		__global float* g_vhat){

	const size_t ty = get_local_id(1),
		tx = get_local_id(0),
		by = get_group_id(1),
		bx = get_group_id(0),
		col = bx*WS+tx,
		row0 = by*WS;


	__local float s_block[ WS ][ WS+1 ];
	float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

	g_in += (row0+ty)*c_width+col;
	g_ybar += by*c_width+col;
	g_vhat += bx*c_height+row0+tx;

#pragma unroll
    for (int i = 0; i < WS -(WS%SOW); i+=SOW) {
        **bdata = *g_in;
        bdata += SOW;
        g_in += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        **bdata = *g_in;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

	if( ty == 0 ) {

        {   // calculate ybar -----------------------
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[0][tx];

            float prev = **bdata;
            ++bdata;

#pragma unroll
            for (int i = 1; i < WS; ++i, ++bdata)
                **bdata = prev = **bdata + prev;

            *g_ybar = prev;
        }

        {   // calculate vhat -----------------------
            float *bdata = s_block[tx];

            float prev = *bdata;
            ++bdata;

#pragma unroll
            for (int i = 1; i < WS; ++i, ++bdata)
                prev = *bdata + prev;

            *g_vhat = prev;
        }

	}
}

//-- Algorithm SAT Stage 2 ----------------------------------------------------

__kernel
void algSAT_stage2(__global float *g_ybar,
                   __global float *g_ysum
                   ) {

	const size_t
			ty = get_local_id(1),
			tx = get_local_id(0),
			bx = get_group_id(0),
			col0 = bx*MW+ty,
			col = col0*WS+tx;

	if( col >= c_width ) return;

	g_ybar += col;
	float y = *g_ybar;
	int ln = HWS+tx;

	if( tx == WS-1 )
		g_ysum += col0;

	volatile __local float s_block[ MW ][ HWS+WS+1 ];

	if( tx < HWS ) s_block[ty][tx] = 0.f;
	else s_block[ty][ln] = 0.f;

	for (int n = 1; n < c_n_size; ++n) {

        // calculate ysum -----------------------

		s_block[ty][ln] = y;

		s_block[ty][ln] += s_block[ty][ln-1];
		s_block[ty][ln] += s_block[ty][ln-2];
		s_block[ty][ln] += s_block[ty][ln-4];
		s_block[ty][ln] += s_block[ty][ln-8];
		s_block[ty][ln] += s_block[ty][ln-16];

		if( tx == WS-1 ) {
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
void algSAT_stage3( __global const float *g_ysum,
		__global float *g_vhat) {

	const size_t tx = get_local_id(0), ty = get_local_id(1),
        by = get_group_id(1), row0 = by*MW+ty, row = row0*WS+tx;

	if( row >= c_height ) return;

	g_vhat += row;
	float y = 0.f, v = 0.f;

	if( row0 > 0 )
		g_ysum += (row0-1)*c_m_size;

	for (int m = 0; m < c_m_size; ++m) {

        // fix vhat -> v -------------------------

		if( row0 > 0 ) {
			y = *g_ysum;
			g_ysum += 1;
		}

		v = *g_vhat += v + y;
		g_vhat += c_height;

	}

}

//-- Algorithm SAT Stage 4 ----------------------------------------------------

__kernel
void algSAT_stage4(__global float *g_inout,
		__global const float *g_y,
		__global const float *g_v ) {

	const size_t tx = get_local_id(0), ty = get_local_id(1),
        bx = get_group_id(0), by = get_group_id(1), col = bx*WS+tx, row0 = by*WS;

	__local float s_block[ WS ][ WS+1 ];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

	g_inout += (row0+ty)*c_width+col;
	if( by > 0 ) g_y += (by-1)*c_width+col;
	if( bx > 0 ) g_v += (bx-1)*c_height+row0+tx;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        **bdata = *g_inout;
        bdata += SOW;
        g_inout += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        **bdata = *g_inout;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

	if( ty == 0 ) {

        {   // calculate y -----------------------
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[0][tx];

            float prev;
            if( by > 0 ) prev = *g_y;
            else prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                **bdata = prev = **bdata + prev;
        }

        {   // calculate x -----------------------
            float *bdata = s_block[tx];

            float prev;
            if( bx > 0 ) prev = *g_v;
            else prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                *bdata = prev = *bdata + prev;
        }

	}

	barrier(CLK_LOCAL_MEM_FENCE);

    bdata = (float (*)[WS+1]) &s_block[ty][tx];

	g_inout -= (WS-(WS%SOW))*c_width;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        *g_inout = **bdata;
        bdata += SOW;
        g_inout += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        *g_inout = **bdata;
    }

}

//-- Algorithm SAT Stage 4 (not-in-place) -------------------------------------

__kernel
void algSAT_stage4(__global float *g_out,
		__global const float *g_in,
		__global const float *g_y,
		__global const float *g_v ) {

	const size_t tx = get_local_id(0), ty = get_local_id(1),
	        bx = get_group_id(0), by = get_group_id(1), col = bx*WS+tx, row0 = by*WS;
	__local float s_block[ WS ][ WS+1 ];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

	g_in += (row0+ty)*c_width+col;
	if( by > 0 ) g_y += (by-1)*c_width+col;
	if( bx > 0 ) g_v += (bx-1)*c_height+row0+tx;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        **bdata = *g_in;
        bdata += SOW;
        g_in += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        **bdata = *g_in;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

	if( ty == 0 ) {

        {   // calculate y -----------------------
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[0][tx];

            float prev;
            if( by > 0 ) prev = *g_y;
            else prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                **bdata = prev = **bdata + prev;
        }

        {   // calculate x -----------------------
            float *bdata = s_block[tx];

            float prev;
            if( bx > 0 ) prev = *g_v;
            else prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                *bdata = prev = *bdata + prev;
        }

	}

	barrier(CLK_LOCAL_MEM_FENCE);

    bdata = (float (*)[WS+1]) &s_block[ty][tx];

	g_out += (row0+ty)*c_width+col;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        *g_out = **bdata;
        bdata += SOW;
        g_out += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        *g_out = **bdata;
    }

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
