/*
 * ClState.h
 *
 *  Created on: Feb 4, 2014
 *      Author: algomorph
 */

#ifndef CLSTATE_HPP_
#define CLSTATE_HPP_

#include <cstdlib>
#include <vector>
#include <string>
#include <CL/cl.hpp>
#include "multithreading.h"


#define MAX_KERNEL_COUNT 16

/*
 *
 */
class CLState {
public:
	cl_context context;
	cl_platform_id platform;
	cl_device_id* devices;
	cl_device_id* execDevices;
	cl_uint deviceCount;
	cl_uint execDeviceCount;
	cl_int errNum;
	bool enableProfiling;
	std::vector<cl_kernel> kernels;
	std::vector<cl_program> programs;
	std::vector<cl_mem> deferredReleaseMem;
	std::vector<cl_command_queue> deferredReleaseQueues;
	CLState(bool enableProfile, bool verbose = false,
			bool log = false);
	virtual ~CLState();
	bool engageBarrier(int numKernels);
	/**
	 * Helper function that will load the OpenCL source program, build and return a handle to that OpenCL kernel
	 */
	cl_program compileOCLProgram(const char* sourcePath,const std::string& options);
private:
	bool engagedBarrier;
	CUTBarrier barrier;
};

#endif /* CLSTATE_HPP_ */
