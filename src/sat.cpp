/*
 * sat.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: algomorph
 */
#include <sat.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <iostream>
#include <oclUtils.h>
#include <ClState.hpp>
#include <gpudefs.h>
#include <boost/lexical_cast.hpp>
namespace sat{

void generateConstants( satConstants& sc,
                     const int& w,
                     const int& h ) {

    sc.width = w;
    sc.height = h;
    sc.m_size = (w+WARP_SIZE-1)/WARP_SIZE;
    sc.n_size = (h+WARP_SIZE-1)/WARP_SIZE;
    sc.last_m = sc.m_size-1;
    sc.last_n = sc.n_size-1;
    sc.border = 0;
    sc.carry_width = sc.m_size*WARP_SIZE;
    sc.carry_height = sc.n_size*WARP_SIZE;
    sc.carry_height = h;
    sc.inv_width = 1.f/(float)w;
    sc.inv_height = 1.f/(float)h;

}

std::string
defineSymbol(const char* name,const std::string& value){
	return " -D " + std::string(name) + "=" + value;
}

std::string
genSatClDefineOptions( const satConstants& sc ) {
	return
	defineSymbol("WIDTH",boost::lexical_cast<std::string>(sc.width)) +
    defineSymbol("HEIGHT",boost::lexical_cast<std::string>(sc.height)) +
    defineSymbol("M_SIZE",boost::lexical_cast<std::string>(sc.m_size)) +
    defineSymbol("N_SIZE",boost::lexical_cast<std::string>(sc.n_size)) +
    defineSymbol("LAST_M",boost::lexical_cast<std::string>(sc.last_m)) +
    defineSymbol("LAST_N",boost::lexical_cast<std::string>(sc.last_n)) +
    defineSymbol("BORDER",boost::lexical_cast<std::string>(sc.border)) +
    defineSymbol("CARRY_WIDTH",boost::lexical_cast<std::string>(sc.carry_width)) +
    defineSymbol("CARRY_HEIGHT",boost::lexical_cast<std::string>(sc.carry_height)) +
    defineSymbol("INV_WIDTH",boost::lexical_cast<std::string>(sc.inv_width)) +
    defineSymbol("INV_HEIGHT",boost::lexical_cast<std::string>(sc.inv_height));
}

void prepareSAT( satConstants& sc,
                     float* d_inout,
                     float* d_ybar,
                     float* d_vhat,
                     float* d_ysum,
                     const float *h_in,
                     const int& w,
                     const int& h ) {


    //d_inout.copy_from( h_in, w, h, algs.width, algs.height );
    //d_ybar.resize( algs.n_size * algs.width );
    //d_vhat.resize( algs.m_size * algs.height );
    //d_ysum.resize( algs.m_size * algs.n_size );

}

void computeSummedAreaTable( float* h_inout,
        const int& w,
        const int& h,
        CLState& state) {
	satConstants sc;
	//float* d_out, *d_ybar, *d_vhat, *d_ysum;
	sc.width = w;
	sc.height = h;
	//pad so the matrix dimensions are multiples of 32
	if( w % WARP_SIZE > 0 ) sc.width += (WARP_SIZE - (w % WARP_SIZE));
	if( h % WARP_SIZE > 0 ) sc.height += (WARP_SIZE - (h % WARP_SIZE));
	generateConstants( sc, sc.width, sc.height );
	std::string defineOptions = genSatClDefineOptions( sc );
	cl_program program = state.compileOCLProgram("sat.cl",defineOptions);
}
} //end namespace sat

int main(int argc, char** argv){
	// start the logs
	shrSetLogFileName ("clsat.log");
	const int in_w = 1024, in_h = 1024;
	float* in_gpu = new float[in_w*in_h];
	std::cout << "[sat2] Generating random input image (" << in_w << "x"
	              << in_h << ") ... " << std::flush;
	for (int i = 0; i < in_w*in_h; ++i)
	        in_gpu[i] = rand() % 256;
	std::cout << "done!\n[sat2] Computing summed-area table in the GPU ... "
	              << std::flush;
	CLState state(true);
	sat::computeSummedAreaTable( in_gpu, in_w, in_h, state);

}
