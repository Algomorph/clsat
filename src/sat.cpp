/*
 * sat.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: algomorph
 */
#include <sat.h>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <iostream>
#include <oclUtils.h>
#include <ClState.hpp>
#include <gpudefs.h>
#include <boost/lexical_cast.hpp>
#include <test.hpp>
namespace sat {

void configureSat(satConfig& config, const int& w, const int& h) {
	//pad so the matrix dimensions are multiples of 32
	config.width = w;
	config.height = h;
	config.carryWidth = w + (w % WARP_SIZE > 0) * (WARP_SIZE - (w % WARP_SIZE));
	config.carryHeight = h
			+ (w % WARP_SIZE > 0) * (WARP_SIZE - (h % WARP_SIZE));
	config.colGroupCount = config.carryWidth / WARP_SIZE;
	config.rowGroupCount = config.carryHeight / WARP_SIZE;
	config.iLastColGroup = config.colGroupCount - 1;
	config.iLastRowGroup = config.rowGroupCount - 1;
	config.inputStride = SCHEDULE_OPTIMIZED_N_WARPS * config.carryWidth;
	config.border = 0;
	config.invWidth = 1.f / (float) w;
	config.invHeight = 1.f / (float) h;
}

#define DEF_SYMBOL(name,value) " -D " + std::string(name) + "=" + value

std::string genSatClDefineOptions(const satConfig& config) {
	return
	DEF_SYMBOL("WIDTH", boost::lexical_cast<std::string>(config.width))
			+ DEF_SYMBOL("HEIGHT", boost::lexical_cast<std::string>(config.height))
			+ DEF_SYMBOL("N_COLUMNS",
					boost::lexical_cast<std::string>(config.colGroupCount))
			+ DEF_SYMBOL("N_ROWS",
					boost::lexical_cast<std::string>(config.rowGroupCount))
			+ DEF_SYMBOL("LAST_M",
					boost::lexical_cast<std::string>(config.iLastColGroup))
			+ DEF_SYMBOL("LAST_N",
					boost::lexical_cast<std::string>(config.iLastRowGroup))
			+ DEF_SYMBOL("BORDER", boost::lexical_cast<std::string>(config.border))
			+ DEF_SYMBOL("CARRY_WIDTH",
					boost::lexical_cast<std::string>(config.carryWidth))
			+ DEF_SYMBOL("CARRY_HEIGHT",
					boost::lexical_cast<std::string>(config.carryHeight))
			+ DEF_SYMBOL("INPUT_STRIDE",
					boost::lexical_cast<std::string>(config.inputStride))
			+ DEF_SYMBOL("INV_WIDTH",
					boost::lexical_cast<std::string>(config.invWidth))
			+ DEF_SYMBOL("INV_HEIGHT",
					boost::lexical_cast<std::string>(config.invHeight));
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

void padMatrixLR(float* origMat, float* paddedMat, const int& width, const int& height,
		const int& carryWidth) {
	const size_t origRowSize = width*sizeof(float);
	for(int iRow = 0; iRow < height; iRow++){
		memcpy(paddedMat,origMat,origRowSize);
		paddedMat += carryWidth;
		origMat += width;
	}
}
void unpadMatrixLR(float* destMat, float* paddedMat, const int& width, const int& height,
		const int& carryWidth) {
	const size_t destRowSize = width*sizeof(float);
	for(int iRow = 0; iRow < height; iRow++){
		memcpy(destMat,paddedMat,destRowSize);
		paddedMat += carryWidth;
		destMat += width;
	}
}

void computeSummedAreaTable(float* inOutMatrix, const int& w, const int& h,
		CLState& state) {
	satConfig config;
	configureSat(config, w, h);
	std::string defineOptions = genSatClDefineOptions(config);
	std::cout << "Define options:\n " << defineOptions << std::endl;
	cl_program program = state.compileOCLProgram("sat.cl", defineOptions);
	cl_int errCode;
	float* preppedInOutMatrix;

	bool paddingEnabled = config.carryWidth != config.width || config.carryHeight != config.height;
	if(paddingEnabled){
		preppedInOutMatrix = new float[config.carryWidth*config.carryHeight]();
		padMatrixLR(inOutMatrix, preppedInOutMatrix, config.width, config.height, config.carryWidth);
	}else{
		preppedInOutMatrix = inOutMatrix;
	}

	const size_t matrixBufferSize = config.carryWidth * config.carryHeight
			* sizeof(float);

	//create matrix buffer and copy the host contents
	cl_mem matrix = clCreateBuffer(state.context,
	CL_MEM_READ_WRITE, matrixBufferSize, NULL, &errCode);

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
	cl_kernel stage1 = clCreateKernel(program, "computeBlockAggregates", &errCode);
	cl_kernel stage2 = clCreateKernel(program, "verticalAggregate", &errCode);
	cl_kernel stage3 = clCreateKernel(program, "horizontalAggregate", &errCode);
	cl_kernel stage4 = clCreateKernel(program, "redistributeSAT",
			&errCode);

	size_t group_size_factor;
	errCode = clGetKernelWorkGroupInfo(stage1, state.devices[0],
	CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
			&group_size_factor, NULL);
	std::cout << "Sugguested work group size multiple: " << group_size_factor
			<< std::endl << std::flush;

	size_t max_kernel_group_size;
	errCode = clGetKernelWorkGroupInfo(stage1, state.devices[0],
	CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_kernel_group_size, NULL);
	std::cout << "Maximum work group size: " << max_kernel_group_size
			<< std::endl << std::flush;

	const int nWm = (config.width + MAX_N_THREADS - 1) / MAX_N_THREADS, nHm =
			(config.height + MAX_N_THREADS - 1) / MAX_N_THREADS;

// load data:
	errCode = clEnqueueWriteBuffer(queue, matrix, CL_TRUE, 0, matrixBufferSize,
			preppedInOutMatrix, 0, NULL, NULL);

//===stage 1===

//debug array & buffer
	const int debugBufCount = WARP_SIZE * (WARP_SIZE + 1);
	size_t debugBufSize = debugBufCount * sizeof(float);
	float* debugArray = new float[debugBufCount];
	for (int iFloat; iFloat < debugBufCount; iFloat++) {
		debugArray[iFloat] = 0.0f;
	}
	//prettyPrintMat(debugArray, WARP_SIZE, WARP_SIZE + 1);

	cl_mem debugBuf = clCreateBuffer(state.context,
	CL_MEM_READ_WRITE, debugBufSize, NULL, &errCode);
	errCode = clEnqueueWriteBuffer(queue, debugBuf, CL_TRUE, 0, debugBufSize,
			debugArray, 0, NULL, NULL);

	if (errCode != CL_SUCCESS) {
		shrLog("clCreateBuffer failed to create buffer with error %d.\n",
				errCode);
	}
	state.deferredReleaseMem.push_back(debugBuf);

	const cl_uint workDim = 2;
	const cl_uint stage1argCount = 4;
	size_t stage1GlobalSize[] = { static_cast<size_t>(config.carryWidth),
			static_cast<size_t>(config.colGroupCount
					* SCHEDULE_OPTIMIZED_N_WARPS) };
	size_t stage1LocalSize[] = { WARP_SIZE, SCHEDULE_OPTIMIZED_N_WARPS };
	cl_mem stage1Args[stage1argCount] = { matrix, yBar, vHat, debugBuf };
	runKernel(stage1, queue, stage1argCount, stage1Args, workDim,
			stage1GlobalSize, stage1LocalSize);
	//get data back from GPU
	//cl_event gpudone_event;
//	errCode = clEnqueueReadBuffer(queue, debugBuf, CL_TRUE, 0, debugBufSize,
//			debugArray, 0, NULL, NULL);

	//prettyPrintMat(debugArray, WARP_SIZE, WARP_SIZE + 1);
	if (errCode != CL_SUCCESS) {
		shrLog("clEnqueueReadBuffer failed to read buffer with error %d.\n",
				errCode);
	}

	//===stage 2===
	//initiate global & group sizes
	const cl_uint stage2argCount = 2;
	size_t stage2GlobalSize[] = { static_cast<size_t>(nWm * WARP_SIZE),
			static_cast<size_t>(MAX_WARPS) };
	size_t stage2LocalSize[] = { WARP_SIZE, MAX_WARPS };
	cl_mem stage2Args[stage2argCount] = { yBar, ySum };
	runKernel(stage2, queue, stage2argCount, stage2Args, workDim,
			stage2GlobalSize, stage2LocalSize);

	//===stage 3===
	//initiate global & group sizes
	const cl_uint stage3argCount = 2;
	size_t stage3GlobalSize[] = { static_cast<size_t>(WARP_SIZE),
			static_cast<size_t>(nHm * MAX_WARPS) };
	size_t stage3LocalSize[] = { WARP_SIZE, MAX_WARPS };
	cl_mem stage3Args[stage3argCount] = { ySum, vHat };
	runKernel(stage3, queue, stage3argCount, stage3Args, workDim,
			stage3GlobalSize, stage3LocalSize);

	//===stage 4===
	//initiate global & group sizes
	const cl_uint stage4argCount = 3;
	size_t stage4GlobalSize[] = { static_cast<size_t>(config.carryWidth),
			static_cast<size_t>(config.colGroupCount
					* SCHEDULE_OPTIMIZED_N_WARPS) };
	size_t stage4LocalSize[] = { WARP_SIZE, SCHEDULE_OPTIMIZED_N_WARPS };
	cl_mem stage4Args[stage4argCount] = { matrix, yBar, vHat };
	runKernel(stage4, queue, stage4argCount, stage4Args, workDim,
			stage4GlobalSize, stage4LocalSize);
	errCode = clFlush(queue);
	if (errCode != CL_SUCCESS) {
		shrLog("clFlush failed to with error %d.\n", errCode);
	}

	//get data back from GPU
	//cl_event gpudone_event;
	errCode = clEnqueueReadBuffer(queue, matrix, CL_TRUE, 0,
			static_cast<size_t>(matrixBufferSize), preppedInOutMatrix, 0, NULL,
			NULL);
	if (errCode != CL_SUCCESS) {
		shrLog("clEnqueueReadBuffer failed to read buffer with error %d.\n",
				errCode);
	}
	if(paddingEnabled){
		unpadMatrixLR(inOutMatrix,preppedInOutMatrix,config.width,config.height,config.carryWidth);
	}
	delete[] debugArray;
} //end computeSummedAreaTable
} //end namespace sat

int main(int argc, char** argv) {
// start the logs
	shrSetLogFileName("clsat.log");
	const int width = 16, height = 16;
	float* inOutMatrix = new float[width * height];
	std::cout << "[clsat] Generating random input image (" << width << "x"
			<< height << ") ... " << std::flush;
	for (int i = 0; i < width * height; ++i)
		inOutMatrix[i] = 1; //rand() % 256;
	std::cout << "done!" << std::endl
			<< "[clsat] Computing summed-area table in the GPU ... "
			<< std::endl << std::flush;
	CLState state(true, true, true);
	sat::computeSummedAreaTable(inOutMatrix, width, height, state);
	prettyPrintMatrix(inOutMatrix, width, height);
	delete[] inOutMatrix;
	return 0;
}

