/*
 * ClState.h
 *
 *  Created on: Feb 4, 2014
 *      Author: algomorph
 */

#ifndef CLSTATE_H_
#define CLSTATE_H_

#include <stdlib.h>
#include <vector>
#include <CL/cl.h>
#include "multithreading.h"

#define MAX_KERNEL_COUNT 16

/*
 *
 */
class ClState {
public:
	cl_context cxGPUContext;
	cl_platform_id cpPlatform;
	cl_device_id* cdDevices;
	cl_device_id* execDevices;
	cl_uint ciDeviceCount;
	cl_int ciErrNum;
	std::vector<cl_kernel> kernels;
	std::vector<cl_program> programs;
	std::vector<cl_mem> vDeferredReleaseMem;
	std::vector<cl_command_queue> vDeferredReleaseQueue;
	ClState(bool bEnableProfile, bool verbose = false,
			bool log = false);
	virtual ~ClState();
	bool engageBarrier(int numKernels);
private:
	bool engagedBarrier;
	CUTBarrier barrier;
};

#endif /* CLSTATE_H_ */
