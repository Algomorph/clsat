/*
 * sat.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: algomorph
 */
#include <sat.h>
#include <oclUtils.h>
#include <ClState.h>
#include <gpudefs.h>
namespace sat{
void algSAT( float* d_out,
             float* d_ybar,
             float* d_vhat,
             float* d_ysum,
             const float* d_in ) {

	//const int nWm = (c_width+MTS-1)/MTS, nHm = (c_height+MTS-1)/MTS;
    //const uint3 cg_img = (uint3)( c_m_size, c_n_size, 1);
    //const uint3 cg_ybar = (uint3)( nWm, 1, 1 );
    //const uint3 cg_vhat = (uint3)( 1, nHm,1 );

    //algSAT_stage1<<< cg_img, dim3(WS, SOW) >>>( d_in, d_ybar, d_vhat );

    //algSAT_stage2<<< cg_ybar, dim3(WS, MW) >>>( d_ybar, d_ysum );

    //algSAT_stage3<<< cg_vhat, dim3(WS, MW) >>>( d_ysum, d_vhat );

    //algSAT_stage4<<< cg_img, dim3(WS, SOW) >>>( d_out, d_in, d_ybar, d_vhat );

}
}//end namespace sat

int main(int argc, char** argv){
	ClState state(true);
}
