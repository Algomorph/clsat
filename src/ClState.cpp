/*
 * ClState.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: algomorph
 */

#include <ClState.hpp>
#include <oclUtils.h>
#include <shrQATest.h>
#include <exception.hpp>
#include <stdio.h>

CLState::CLState(bool enableProfile, bool verbose, bool log) :
		platform(NULL), devices(NULL), execDevices(NULL), deviceCount(0), execDeviceCount(
				0), errNum(CL_SUCCESS), enableProfiling(enableProfile), engagedBarrier(false) {

	cl_int ciErrNum = CL_SUCCESS;

	//Get the NVIDIA platform
	ciErrNum = oclGetPlatformID(&this->platform);
	if (ciErrNum != CL_SUCCESS && log) {
		shrLog("Error: Failed to create OpenCL context!\n");
	} else {
		//Retrieve available OpenCL GPU devices
		ciErrNum = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU, 0, NULL,
				&this->deviceCount);

		this->devices = (cl_device_id *) malloc(
				this->deviceCount * sizeof(cl_device_id));
		ciErrNum = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU,
				this->deviceCount, this->devices, NULL);
		std::cout << this->deviceCount << std::endl << std::flush;
		// Allocate a buffer array to store the names GPU device(s)
		char (*clDeviceNames)[256] = new char[this->deviceCount][256];
		if (ciErrNum != CL_SUCCESS && log) {
			shrLog("Error: Failed to create OpenCL context!\n");
		} else {
			if (log) {
				shrLog(
						"Detected %d OpenCL devices of type CL_DEVICE_TYPE_GPU\n",
						this->deviceCount);
			}
			for (int i = 0; i < (int) this->deviceCount; i++) {
				cl_ulong globalMemSize = 0;
				cl_ulong localMemSize = 0;
				cl_uint maxComputeUnits = 0;
				size_t maxWorkGroupSize = 0;

				clGetDeviceInfo(this->devices[i], CL_DEVICE_NAME,
						sizeof(clDeviceNames[i]), &clDeviceNames[i], NULL);
				clGetDeviceInfo(this->devices[i], CL_DEVICE_GLOBAL_MEM_SIZE,
						sizeof(clDeviceNames[i]), &globalMemSize, NULL);
				clGetDeviceInfo(this->devices[i], CL_DEVICE_LOCAL_MEM_SIZE,
						sizeof(clDeviceNames[i]), &localMemSize, NULL);
				clGetDeviceInfo(this->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS,
						sizeof(clDeviceNames[i]), &maxComputeUnits, NULL);
				clGetDeviceInfo(this->devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE,
						sizeof(clDeviceNames[i]), &maxWorkGroupSize, NULL);

				if (log) {
					shrLog("> OpenCL Device #%d (%s), cl_device_id: %d\n"
							"  Device global memory: %d bytes\n"
							"  Device local memory: %d bytes\n"
							"  Device max compute units: %d\n"
							"  Device max work-group size: %d\n", i,
							clDeviceNames[i], this->devices[i], globalMemSize,
							localMemSize, maxComputeUnits, maxWorkGroupSize);
				}
			}
			//Create the OpenCL context
			this->context = clCreateContext(0, this->deviceCount, this->devices,
			NULL, NULL, &ciErrNum);
			if (ciErrNum != CL_SUCCESS && log) {
				shrLog("Error: Failed to create OpenCL context!\n");
			}
		}
		delete[] clDeviceNames;

	}
	this->errNum = ciErrNum;
}

bool CLState::engageBarrier(int numKernels) {
	if (!this->engagedBarrier) {
		this->engagedBarrier = true;
		this->barrier = cutCreateBarrier(numKernels);
		return true;
	} else {
		return false;
	}
}

/**
 * Helper function that will load the OpenCL source program, build and return a handle to that OpenCL kernel
 * @param context - the OpenCL context
 * @param device - the device to compile for
 * @param program - the program that is being built (in/out)
 * @param sourcePath - full path to the source file
 * @param options - build options, e.g. define flags("-D name=value")
 * @return an error code on failure, 0 on success
 */
cl_program CLState::compileOCLProgram(const char* sourcePath,
		const std::string& options) {
	cl_int errNum;

	size_t program_length;
	oclCheckError(sourcePath != NULL, shrTRUE);
	char *source = oclLoadProgSource(sourcePath, "", &program_length);
	if (!source) {
		shrLog("Error: Failed to load compute program %s!\n", sourcePath);
		BOOST_THROW_EXCEPTION(
				runtime_error()
						<< error_message(
								(std::string(
										"Error: Failed to load cl program source from ")
										+ sourcePath).c_str()));
	}

	// create the simple increment OpenCL program
	cl_program program = clCreateProgramWithSource(context, 1,
			(const char **) &source, &program_length, &errNum);
	if (errNum != CL_SUCCESS) {
		shrLog("Error: Failed to create program\n");
		BOOST_THROW_EXCEPTION(runtime_error() << cl_error_code(errNum));
	} else {
		shrLog("clCreateProgramWithSource <%s> succeeded, program_length=%d\n",
				sourcePath, program_length);
	}
	free(source);

	// build the program
	cl_build_status build_status = CL_SUCCESS;

	errNum =
			clBuildProgram(program, 0, NULL,
					std::string(
							"-cl-fast-relaxed-math -cl-nv-verbose" + options).c_str(),
					NULL, NULL);
	if (errNum != CL_SUCCESS) {
		// write out standard error, Build Log and PTX, then return error
		shrLogEx(LOGBOTH | ERRORMSG, errNum, STDERROR);
		oclLogBuildInfo(program, oclGetFirstDev(context));
		oclLogPtx(program, oclGetFirstDev(context), "build_error.ptx");
		BOOST_THROW_EXCEPTION(runtime_error() << cl_error_code(errNum));
	} else {
		shrLog("clBuildProgram <%s> succeeded\n", sourcePath);
		if (this->execDevices != NULL) {
			for (uint iDevice = 0;
					iDevice < this->execDeviceCount
							&& build_status == CL_SUCCESS
							&& errNum == CL_SUCCESS; iDevice++) {
				std::cout << "Hello!" << std::endl;
				cl_device_id device = this->execDevices[iDevice];
				errNum = clGetProgramBuildInfo(program, device,
				CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status,
				NULL);
				shrLog("clGetProgramBuildInfo returned: ");
				if (build_status == CL_SUCCESS) {
					shrLog("CL_SUCCESS\n");
				} else {
					shrLog("CLErrorNumber = %d\n", errNum);
				}
				// print out the build log, note in the case where there is nothing shown, some OpenCL PTX->SASS caching has happened
				{
					char *build_log;
					size_t ret_val_size;

					errNum = clGetProgramBuildInfo(program, device,
					CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

					if (errNum != CL_SUCCESS) {
						shrLog(
								"clGetProgramBuildInfo device %d, failed to get the log size at line %d\n",
								device, __LINE__);
					}
					build_log = (char *) malloc(ret_val_size + 1);

					errNum = clGetProgramBuildInfo(program, device,
					CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

					if (errNum != CL_SUCCESS) {
						shrLog(
								"clGetProgramBuildInfo device %d, failed to get the build log at line %d\n",
								device, __LINE__);
					}
					// to be carefully, terminate with \0
					// there's no information in the reference whether the string is 0 terminated or not
					build_log[ret_val_size] = '\0';
					shrLog("%s\n", build_log);
				}
			}
		}
	}
	this->programs.push_back(program);
	return program;
}

CLState::~CLState() {
	clReleaseContext(this->context);
	if (this->engagedBarrier) {
		cutWaitForBarrier(&this->barrier);
	}
	for (cl_mem memObject : this->deferredReleaseMem) {
		clReleaseMemObject(memObject);
	}
	for (cl_command_queue queue : this->deferredReleaseQueues) {
		clReleaseCommandQueue(queue);
	}
	for (cl_kernel kernel : this->kernels) {
		clReleaseKernel(kernel);
	}
	for (cl_program program : this->programs) {
		clReleaseProgram(program);
	}
}

