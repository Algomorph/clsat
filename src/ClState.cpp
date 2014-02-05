/*
 * ClState.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: algomorph
 */

#include <ClState.h>
#include <oclUtils.h>
#include <shrQATest.h>
#include <CL/cl.h>


ClState::ClState(bool bEnableProfile, bool verbose, bool log) :
		cpPlatform(NULL), cdDevices(NULL), execDevices(NULL), ciErrNum(
		CL_SUCCESS), ciDeviceCount(0), engagedBarrier(false) {

	cl_int ciErrNum = CL_SUCCESS;

	//Get the NVIDIA platform
	ciErrNum = oclGetPlatformID(&this->cpPlatform);
	if (ciErrNum != CL_SUCCESS && log) {
		shrLog("Error: Failed to create OpenCL context!\n");
	} else {
		//Retrieve available OpenCL GPU devices
		ciErrNum = clGetDeviceIDs(this->cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL,
				&this->ciDeviceCount);
		this->cdDevices = (cl_device_id *) malloc(
				this->ciDeviceCount * sizeof(cl_device_id));
		ciErrNum = clGetDeviceIDs(this->cpPlatform, CL_DEVICE_TYPE_GPU,
				this->ciDeviceCount, this->cdDevices, NULL);
		// Allocate a buffer array to store the names GPU device(s)
		char (*clDeviceNames)[256] = new char[this->ciDeviceCount][256];
		if (ciErrNum != CL_SUCCESS && log) {
			shrLog("Error: Failed to create OpenCL context!\n");
		} else {
			if (log) {
				shrLog(
						"Detected %d OpenCL devices of type CL_DEVICE_TYPE_GPU\n",
						this->ciDeviceCount);
			}
			for (int i = 0; i < (int) this->ciDeviceCount; i++) {
				clGetDeviceInfo(this->cdDevices[i], CL_DEVICE_NAME,
						sizeof(clDeviceNames[i]), &clDeviceNames[i], NULL);
				if (log) {
					shrLog("> OpenCL Device #%d (%s), cl_device_id: %d\n", i,
							clDeviceNames[i], this->cdDevices[i]);
				}
			}
			//Create the OpenCL context
			this->cxGPUContext = clCreateContext(0, this->ciDeviceCount,
					this->cdDevices, NULL, NULL, &ciErrNum);
			if (ciErrNum != CL_SUCCESS && log) {
				shrLog("Error: Failed to create OpenCL context!\n");
			}
		}
		delete[] clDeviceNames;

	}
	this->ciErrNum = ciErrNum;
}

bool
ClState::engageBarrier(int numKernels){
	if(!this->engagedBarrier){
		this->engagedBarrier = true;
		this->barrier = cutCreateBarrier(numKernels);
		return true;
	}else{
		return false;
	}
}

ClState::~ClState() {
	clReleaseContext (this->cxGPUContext);
	if(this->engagedBarrier){
		cutWaitForBarrier(&this->barrier);
	}
	for (cl_mem memObject : this->vDeferredReleaseMem){
		clReleaseMemObject(memObject);
	}
	for (cl_command_queue queue : this->vDeferredReleaseQueue){
		clReleaseCommandQueue(queue);
	}
	for (cl_kernel kernel : this->kernels){
		clReleaseKernel(kernel);
	}
	for (cl_program program : this->programs){
		clReleaseProgram(program);
	}
}

