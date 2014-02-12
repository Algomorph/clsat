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

void generateConstants(satConstants& config, const int& w, const int& h,
		const int& warpSize) {

	config.width = w;
	config.height = h;
	config.colGroupCount = (w + WARP_SIZE - 1) / WARP_SIZE;
	config.rowGroupCount = (h + WARP_SIZE - 1) / WARP_SIZE;
	config.iLastColGroup = config.colGroupCount - 1;
	config.iLastRowGroup = config.rowGroupCount - 1;
	config.border = 0;
	config.carryWidth = config.colGroupCount * WARP_SIZE;
	config.carryHeight = config.rowGroupCount * WARP_SIZE;
	//sc.carryHeight = h;
	config.invWidth = 1.f / (float) w;
	config.invHeight = 1.f / (float) h;

}

#define DEF_SYMBOL(name,value) " -D " + std::string(name) + "=" + value

std::string genSatClDefineOptions(const satConstants& sc) {
	return
	DEF_SYMBOL("WIDTH", boost::lexical_cast<std::string>(sc.width))
			+ DEF_SYMBOL("HEIGHT", boost::lexical_cast<std::string>(sc.height))
			+ DEF_SYMBOL("M_SIZE",
					boost::lexical_cast<std::string>(sc.colGroupCount))
			+ DEF_SYMBOL("N_SIZE",
					boost::lexical_cast<std::string>(sc.rowGroupCount))
			+ DEF_SYMBOL("LAST_M",
					boost::lexical_cast<std::string>(sc.iLastColGroup))
			+ DEF_SYMBOL("LAST_N",
					boost::lexical_cast<std::string>(sc.iLastRowGroup))
			+ DEF_SYMBOL("BORDER", boost::lexical_cast<std::string>(sc.border))
			+ DEF_SYMBOL("CARRY_WIDTH",
					boost::lexical_cast<std::string>(sc.carryWidth))
			+ DEF_SYMBOL("CARRY_HEIGHT",
					boost::lexical_cast<std::string>(sc.carryHeight))
			+ DEF_SYMBOL("INV_WIDTH",
					boost::lexical_cast<std::string>(sc.invWidth))
			+ DEF_SYMBOL("INV_HEIGHT",
					boost::lexical_cast<std::string>(sc.invHeight));
}

void prepareSAT(satConstants& sc, float* d_inout, float* d_ybar, float* d_vhat,
		float* d_ysum, const float *h_in, const int& w, const int& h) {

	//d_inout.copy_from( h_in, w, h, algs.width, algs.height );
	//d_ybar.resize( algs.n_size * algs.width );
	//d_vhat.resize( algs.m_size * algs.height );
	//d_ysum.resize( algs.m_size * algs.n_size );

}

void computeSummedAreaTable(float* inputMatrix, const int& w, const int& h,
		CLState& state) {
	satConstants config;
	config.width = w;
	config.height = h;

	//pad so the matrix dimensions are multiples of 32
	if (w % WARP_SIZE > 0)
		config.width += (WARP_SIZE - (w % WARP_SIZE));
	if (h % WARP_SIZE > 0)
		config.height += (WARP_SIZE - (h % WARP_SIZE));

	generateConstants(config, config.width, config.height);
	std::string defineOptions = genSatClDefineOptions(config);
	cl_program program = state.compileOCLProgram("sat.cl", defineOptions);
	cl_int errCode;

	cl_mem matrix = clCreateBuffer(state.context,
	CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			config.width * config.height * sizeof(float), inputMatrix,
			&errCode);

	cl_mem yBar = clCreateBuffer(state.context, CL_MEM_READ_WRITE,
			config.rowGroupCount * config.width * sizeof(float), NULL,
			&errCode);

	cl_mem vHat = clCreateBuffer(state.context, CL_MEM_READ_WRITE,
			config.colGroupCount * config.height * sizeof(float), NULL,
			&errCode);

	cl_mem ySum = clCreateBuffer(state.context, CL_MEM_READ_WRITE,
			config.rowGroupCount * config.colGroupCount * sizeof(float), NULL,
			&errCode);

	cl_command_queue queue = clCreateCommandQueue(state.context,
			state.devices[0],
			(state.enableProfiling ? CL_QUEUE_PROFILING_ENABLE : 0), &errCode);

	state.deferredReleaseQueues.push_back(queue);
	state.deferredReleaseMem.push_back(matrix);
	state.deferredReleaseMem.push_back(yBar);
	state.deferredReleaseMem.push_back(vHat);
	state.deferredReleaseMem.push_back(ySum);

	cl_kernel stage1 = clCreateKernel(program, "algSAT_stage1", &errCode);
	cl_kernel stage2 = clCreateKernel(program, "algSAT_stage2", &errCode);
	cl_kernel stage3 = clCreateKernel(program, "algSAT_stage3", &errCode);
	cl_kernel stage4 = clCreateKernel(program, "algSAT_stage4_inplace",
			&errCode);

	size_t group_size_factor;
	errCode = clGetKernelWorkGroupInfo(stage1, state.devices[0],
	CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
			&group_size_factor, NULL);
	std::cout << "Sugguested group size: " << group_size_factor << std::endl
			<< std::flush;

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

