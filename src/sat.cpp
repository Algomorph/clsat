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

void generateConstants(satConstants& config, const int& w, const int& h) {
	//pad so the matrix dimensions are multiples of 32
	config.width = w;
	config.height = h;
	config.carryWidth = w + (w % WARP_SIZE > 0)*(WARP_SIZE - (w % WARP_SIZE));
	config.carryHeight = h + (w % WARP_SIZE > 0)*(WARP_SIZE - (h % WARP_SIZE));
	config.colGroupCount = config.carryWidth / WARP_SIZE;
	config.rowGroupCount = config.carryHeight / WARP_SIZE;
	config.iLastColGroup = config.colGroupCount - 1;
	config.iLastRowGroup = config.rowGroupCount - 1;
	config.border = 0;
	config.invWidth = 1.f / (float) w;
	config.invHeight = 1.f / (float) h;
}

#define DEF_SYMBOL(name,value) " -D " + std::string(name) + "=" + value

std::string genSatClDefineOptions(const satConstants& sc) {
	return
	DEF_SYMBOL("WIDTH", boost::lexical_cast<std::string>(sc.width))
			+ DEF_SYMBOL("HEIGHT", boost::lexical_cast<std::string>(sc.height))
			+ DEF_SYMBOL("N_COLUMNS",
					boost::lexical_cast<std::string>(sc.colGroupCount))
			+ DEF_SYMBOL("N_ROWS",
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

void runKernel(const cl_kernel& kernel, const cl_command_queue& queue,
		const cl_uint& argCount, cl_mem* args, const cl_uint& dim,
		size_t* globalSize, size_t* localSize) {
	cl_int errCode = CL_SUCCESS;
	for (cl_uint iMem = 0; iMem < argCount; iMem++) {
		errCode = clSetKernelArg(kernel, iMem, sizeof(cl_mem),
				(void *) (args + iMem));
		if (errCode != CL_SUCCESS) {
			shrLog("clSetKernelArg failed to set argument with error %d.\n",
					errCode);
		}
	}
	errCode = clEnqueueNDRangeKernel(queue, kernel, dim, 0, globalSize,
			localSize, 0, 0, 0);
	if (errCode != CL_SUCCESS) {
		shrLog(
				"clEnqueueNDRangeKernel failed to enqueue the kernel with error %d.\n",
				errCode);
	}
}

void computeSummedAreaTable(float* inOutMatrix, const int& w, const int& h,
		CLState& state) {
	satConstants config;
	generateConstants(config, w, h);
	std::string defineOptions = genSatClDefineOptions(config);
	std::cout << "Define options:\n " << defineOptions << std::endl;
	cl_program program = state.compileOCLProgram("sat.cl", defineOptions);
	cl_int errCode;

	size_t matrixBufferSize = config.carryWidth * config.carryHeight * sizeof(float);

	//create matrix buffer and copy the host contents
	cl_mem matrix = clCreateBuffer(state.context,
	CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrixBufferSize, inOutMatrix,
			&errCode);
	if (errCode != CL_SUCCESS) {
		shrLog(
				"Failed to create matrix buffer & copy the contents to device, error %d.\n",
				errCode);
	}

	//create buffers in GPU memory for intermediate results
	cl_mem yBar = clCreateBuffer(state.context, CL_MEM_READ_WRITE,
			config.rowGroupCount * config.carryWidth * sizeof(float), NULL,
			&errCode);
	if (errCode != CL_SUCCESS) {
		shrLog("Failed to create buffer, error %d.\n", errCode);
	}

	cl_mem vHat = clCreateBuffer(state.context, CL_MEM_READ_WRITE,
			config.colGroupCount * config.carryHeight * sizeof(float), NULL,
			&errCode);
	if (errCode != CL_SUCCESS) {
		shrLog("Failed to create buffer, error %d.\n", errCode);
	}

	cl_mem ySum = clCreateBuffer(state.context, CL_MEM_READ_WRITE,
			config.rowGroupCount * config.colGroupCount * sizeof(float), NULL,
			&errCode);
	if (errCode != CL_SUCCESS) {
		shrLog("Failed to create buffer, error %d.\n", errCode);
	}

	cl_command_queue queue = clCreateCommandQueue(state.context,
			state.devices[0],
			(state.enableProfiling ? CL_QUEUE_PROFILING_ENABLE : 0), &errCode);

	//push back elements to be released after the program has run
	state.deferredReleaseQueues.push_back(queue);
	state.deferredReleaseMem.push_back(matrix);
	state.deferredReleaseMem.push_back(yBar);
	state.deferredReleaseMem.push_back(vHat);
	state.deferredReleaseMem.push_back(ySum);

	//create kernels
	cl_kernel testKernel = clCreateKernel(program, "testKernel", &errCode);
	cl_kernel stage1 = clCreateKernel(program, "algSAT_stage1", &errCode);
	cl_kernel stage2 = clCreateKernel(program, "algSAT_stage2", &errCode);
	cl_kernel stage3 = clCreateKernel(program, "algSAT_stage3", &errCode);
	cl_kernel stage4 = clCreateKernel(program, "algSAT_stage4_inplace",
			&errCode);

	size_t group_size_factor;
	errCode = clGetKernelWorkGroupInfo(stage1, state.devices[0],
	CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
			&group_size_factor, NULL);
	std::cout << "Sugguested work group size multiple: " << group_size_factor << std::endl
				<< std::flush;

	size_t max_kernel_group_size;
	errCode = clGetKernelWorkGroupInfo(stage1, state.devices[0],
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
				&max_kernel_group_size, NULL);
	std::cout << "Maximum work group size: " << max_kernel_group_size << std::endl
					<< std::flush;

	const int nWm = (config.width + MAX_N_THREADS - 1) / MAX_N_THREADS, nHm =
			(config.height + MAX_N_THREADS - 1) / MAX_N_THREADS;

	//test

	const cl_uint testArgCount = 1;
	size_t testGlobalSize[] = { static_cast<size_t>(config.carryWidth),
				static_cast<size_t>(config.carryHeight)};
	size_t testLocalSize[] = {WARP_SIZE, WARP_SIZE};
	cl_mem testArgs[testArgCount] = {matrix};
//	runKernel(testKernel, queue, testArgCount, testArgs, 2, testGlobalSize, testLocalSize);

	//===stage 1===
	const cl_uint workDim = 2;
	const cl_uint stage1argCount = 3;
	size_t stage1GlobalSize[] = { static_cast<size_t>(config.carryWidth),
			static_cast<size_t>(config.colGroupCount
					* SCHEDULE_OPTIMIZED_N_WARPS) };
	size_t stage1LocalSize[] = { WARP_SIZE, SCHEDULE_OPTIMIZED_N_WARPS };
	cl_mem stage1Args[stage1argCount] = { matrix, yBar, vHat };
	runKernel(stage1, queue, stage1argCount, stage1Args, workDim,
			stage1GlobalSize, stage1LocalSize);

	//===stage 2===
	//initiate global & group sizes
	const cl_uint stage2argCount = 2;
	size_t stage2GlobalSize[] = { static_cast<size_t>(nWm * WARP_SIZE),
			static_cast<size_t>(MAX_WARPS) };
	size_t stage2LocalSize[] = { WARP_SIZE, MAX_WARPS };
	cl_mem stage2Args[stage2argCount] = { yBar, ySum };
//	runKernel(stage2, queue, stage2argCount, stage2Args, workDim,
//			stage2GlobalSize, stage2LocalSize);

	//===stage 3===
	//initiate global & group sizes
	const cl_uint stage3argCount = 2;
	size_t stage3GlobalSize[] = { static_cast<size_t>(WARP_SIZE),
			static_cast<size_t>(nHm * MAX_WARPS) };
	size_t stage3LocalSize[] = { WARP_SIZE, MAX_WARPS };
	cl_mem stage3Args[stage3argCount] = { ySum, vHat };
//	runKernel(stage3, queue, stage3argCount, stage3Args, workDim,
//			stage3GlobalSize, stage3LocalSize);

	//===stage 4===
	//initiate global & group sizes
	const cl_uint stage4argCount = 3;
	size_t stage4GlobalSize[] = { static_cast<size_t>(config.carryWidth),
			static_cast<size_t>(config.colGroupCount
					* SCHEDULE_OPTIMIZED_N_WARPS) };
	size_t stage4LocalSize[] = { WARP_SIZE, SCHEDULE_OPTIMIZED_N_WARPS };
	cl_mem stage4Args[stage4argCount] = { matrix, yBar, vHat };
//	runKernel(stage4, queue, stage4argCount, stage4Args, workDim,
//			stage4GlobalSize, stage4LocalSize);
	errCode = clFlush(queue);
	if (errCode != CL_SUCCESS) {
		shrLog("clFlush failed to with error %d.\n", errCode);
	}

	//get data back from GPU
	//cl_event gpudone_event;
	errCode = clEnqueueReadBuffer(queue, matrix, CL_TRUE, 0,
			static_cast<size_t>(w * h * sizeof(float)), inOutMatrix, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		shrLog("clEnqueueReadBuffer failed to read buffer with error %d.\n",
				errCode);
	}
} //end computeSummedAreaTable
} //end namespace sat

int main(int argc, char** argv) {
// start the logs
	shrSetLogFileName("clsat.log");
	const int in_w = 1024, in_h = 1024;
	//const int in_w = 10, in_h = 10;
	float* inOutMatrix = new float[in_w * in_h];
	std::cout << "[clsat] Generating random input image (" << in_w << "x"
			<< in_h << ") ... " << std::flush;
	for (int i = 0; i < in_w * in_h; ++i)
		inOutMatrix[i] = rand() % 256;
	std::cout << "done!" << std::endl
			<< "[clsat] Computing summed-area table in the GPU ... "
			<< std::endl << std::flush;
	CLState state(true, true, true);
	sat::computeSummedAreaTable(inOutMatrix, in_w, in_h, state);

}

