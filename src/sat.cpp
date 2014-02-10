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
namespace sat {

void generateConstants(satConstants& sc, const int& w, const int& h) {

	sc.width = w;
	sc.height = h;
	sc.m_size = (w + WARP_SIZE - 1) / WARP_SIZE;
	sc.n_size = (h + WARP_SIZE - 1) / WARP_SIZE;
	sc.last_m = sc.m_size - 1;
	sc.last_n = sc.n_size - 1;
	sc.border = 0;
	sc.carry_width = sc.m_size * WARP_SIZE;
	sc.carry_height = sc.n_size * WARP_SIZE;
	sc.carry_height = h;
	sc.inv_width = 1.f / (float) w;
	sc.inv_height = 1.f / (float) h;

}

#define DEF_SYMBOL(name,value) " -D " + std::string(name) + "=" + value

std::string genSatClDefineOptions(const satConstants& sc) {
	return
	DEF_SYMBOL("WIDTH", boost::lexical_cast<std::string>(sc.width))
			+ DEF_SYMBOL("HEIGHT", boost::lexical_cast<std::string>(sc.height))
			+ DEF_SYMBOL("M_SIZE", boost::lexical_cast<std::string>(sc.m_size))
			+ DEF_SYMBOL("N_SIZE", boost::lexical_cast<std::string>(sc.n_size))
			+ DEF_SYMBOL("LAST_M", boost::lexical_cast<std::string>(sc.last_m))
			+ DEF_SYMBOL("LAST_N", boost::lexical_cast<std::string>(sc.last_n))
			+ DEF_SYMBOL("BORDER", boost::lexical_cast<std::string>(sc.border))
			+ DEF_SYMBOL("CARRY_WIDTH",
					boost::lexical_cast<std::string>(sc.carry_width))
			+ DEF_SYMBOL("CARRY_HEIGHT",
					boost::lexical_cast<std::string>(sc.carry_height))
			+ DEF_SYMBOL("INV_WIDTH",
					boost::lexical_cast<std::string>(sc.inv_width))
			+ DEF_SYMBOL("INV_HEIGHT",
					boost::lexical_cast<std::string>(sc.inv_height));
}

void prepareSAT(satConstants& sc, float* d_inout, float* d_ybar, float* d_vhat,
		float* d_ysum, const float *h_in, const int& w, const int& h) {

	//d_inout.copy_from( h_in, w, h, algs.width, algs.height );
	//d_ybar.resize( algs.n_size * algs.width );
	//d_vhat.resize( algs.m_size * algs.height );
	//d_ysum.resize( algs.m_size * algs.n_size );

}

void computeSummedAreaTable(float* h_inout, const int& w, const int& h,
		CLState& state) {
	satConstants sc;
	float* d_out, *d_ybar, *d_vhat, *d_ysum;
	sc.width = w;
	sc.height = h;
	//pad so the matrix dimensions are multiples of 32
	if (w % WARP_SIZE > 0)
		sc.width += (WARP_SIZE - (w % WARP_SIZE));
	if (h % WARP_SIZE > 0)
		sc.height += (WARP_SIZE - (h % WARP_SIZE));
	generateConstants(sc, sc.width, sc.height);
	d_out = new float[sc.width * sc.height];
	std::string defineOptions = genSatClDefineOptions(sc);
	cl_program program = state.compileOCLProgram("sat.cl", defineOptions);
	cl_int errCode;
	cl_kernel stage1 = clCreateKernel(program, "algSAT_stage1", &errCode);
	cl_kernel stage2 = clCreateKernel(program, "algSAT_stage2", &errCode);
	cl_kernel stage3 = clCreateKernel(program, "algSAT_stage3", &errCode);
	cl_kernel stage4 = clCreateKernel(program, "algSAT_stage4_inplace",
			&errCode);
	size_t group_size_factor;
	errCode = clGetKernelWorkGroupInfo(stage1, state.devices[0],
	CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
			&group_size_factor, NULL);
	std::cout << group_size_factor << std::endl << std::flush;
}
} //end namespace sat

int main(int argc, char** argv) {
	// start the logs
	shrSetLogFileName("clsat.log");
	const int in_w = 1024, in_h = 1024;
	float* in_gpu = new float[in_w * in_h];
	std::cout << "[sat2] Generating random input image (" << in_w << "x" << in_h
			<< ") ... " << std::flush;
	for (int i = 0; i < in_w * in_h; ++i)
		in_gpu[i] = rand() % 256;
	std::cout << "done!" << std::endl
			<< "[sat2] Computing summed-area table in the GPU ... " << std::endl
			<< std::flush;
	CLState state(true, true, true);
	sat::computeSummedAreaTable(in_gpu, in_w, in_h, state);

}

