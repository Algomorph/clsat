/*
 * Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample shows the implementation of multi-threaded heterogeneous computing workloads with tight cooperation between CPU and GPU.
 * With OpenCL 1.1 the API introduces three new concepts that are utilized:
 * 1) User Events
 * 2) Thread-Safe API calls
 * 3) Event Callbacks
 *
 * The workloads in the sample follow the form CPU preprocess -> GPU process -> CPU postprocess.
 * Each CPU processing step is handled by its own dedicated thread. GPU workloads are sent to all available GPUs in the system.
 *
 * A user event is used to stall enqueued GPU work until the CPU has finished the preprocessing. Preprocessing is
 * handled by a dedicated CPU thread and relies on thread-safe API calls to signal the GPU that the main processing
 * can start. The new event callback mechanism of OpenCL is used to launch a new CPU thread on event completion of
 * downloading data from GPU.
 */

#include <stdio.h>
#include <shrQATest.h>
#include <oclUtils.h>
#include <vector>
#include <ClState.hpp>

#ifdef _WIN32
#include <windows.h>
const char *getOSNameWindows(OSVERSIONINFO *pOSVI)
{
	pOSVI->dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
	GetVersionEx (pOSVI);

	if ((pOSVI->dwMajorVersion ==6)&&(pOSVI->dwMinorVersion==1))
	return (const char *)"Windows 7";
	else if ((pOSVI->dwMajorVersion ==6)&&(pOSVI->dwMinorVersion==0))
	return (const char *)"Windows Vista";
	else if ((pOSVI->dwMajorVersion ==5)&&(pOSVI->dwMinorVersion==1))
	return (const char *)"Windows XP";
	else if ((pOSVI->dwMajorVersion ==5)&&(pOSVI->dwMinorVersion==0))
	return (const char *)"Windows 2000";
	else if ((pOSVI->dwMajorVersion ==4)&&(pOSVI->dwMinorVersion==0))
	return (const char *)"Windows NT 4.0";
	else if ((pOSVI->dwMajorVersion ==3)&&(pOSVI->dwMinorVersion==51))
	return (const char *)"Windows NT 3.51";
	else if ((pOSVI->dwMajorVersion ==4)&&(pOSVI->dwMinorVersion==90))
	return (const char *)"Windows ME";
	else if ((pOSVI->dwMajorVersion ==4)&&(pOSVI->dwMinorVersion==10))
	return (const char *)"Windows 98";
	else if((pOSVI->dwMajorVersion ==4)&&(pOSVI->dwMinorVersion==0))
	return (const char *)"Windows 95";
	else
	return (const char *)"Windows OS Unknown";
}
#else
const char *getOSNameUnix() {
	return (const char *) "UNIX Operating System";
}
#endif

const char* getOSName() {
#ifdef _WIN32
	// we are detecting what Windows OS is being used, as the Windows threading types requires Windows Vista/7
	OSVERSIONINFO OSversion;
	return getOSName(&OSversion));
#else
	return getOSNameUnix();
#endif
}

#include "multithreading.h"

const int N = 8;
const int buffer_size = 1 << 23;
const int BLOCK_SIZE = 16;
const int MAX_GPU_COUNT = 16;

#ifdef _WIN32
#define CALLBACK_FUNC void __stdcall

#else
#define CALLBACK_FUNC void
#endif


/*TODO: figure out if we still need this for hybrid workloads
struct cpu_worker_arg_t {
	int* data_n;
	float* data_fp;
	cl_event user_event;
	int id;
	bool bEnableProfile;
};


bool bOK = true;


// First part of the heterogeneous workload, prcoessing done by CPU
CUT_THREADPROC cpu_preprocess(void* void_arg) {
	cpu_worker_arg_t* arg = (cpu_worker_arg_t*) void_arg;

	for (int i = 0; i < buffer_size / sizeof(int); ++i) {
		arg->data_fp[i] = (float) arg->id;
	}

	// Signal GPU that CPU is done preprocessing via OpenCL user event
	clSetUserEventStatus(arg->user_event, CL_COMPLETE);
	CUT_THREADEND;
}

// last part of the heterogeneous workload, processing done by CPU
CUT_THREADPROC cpu_postprocess(void* void_arg) {
	cpu_worker_arg_t* arg = (cpu_worker_arg_t*) void_arg;

	for (int i = 0; i < buffer_size / sizeof(int) && bOK; ++i) {
		if (arg->data_fp[i] != (float) arg->id + 1.0f) {
			bOK = false;
			shrLog("Results don't match in workload %d!\n", arg->id);
		}
	}

	// Cleanup
	free(arg->data_fp);
	free(void_arg);

	// Signal that this job has finished
	cutIncrementBarrier(&barrier);
	CUT_THREADEND;
}

CALLBACK_FUNC event_callback(cl_event event, cl_int event_command_exec_status,
		void* user_data) {
	if (event_command_exec_status != CL_COMPLETE) {
		shrLog("clEnqueueWriteBuffer() Error: Failed to write buffer!\n");
		cutIncrementBarrier(&barrier);
		return;
	}

	// Profile the OpenCL kernel event information
	cl_ulong ev_start_time = (cl_ulong) 0;
	cl_ulong ev_end_time = (cl_ulong) 0;
	size_t return_bytes;

	if (((cpu_worker_arg_t *) user_data)->bEnableProfile == true) {
		int err;
		err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
				sizeof(cl_ulong), &ev_start_time, &return_bytes);

		err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
				sizeof(cl_ulong), &ev_end_time, &return_bytes);

		double run_time = (double) (ev_end_time - ev_start_time);
		printf("\t> event_callback() event_id=%d, kernel runtime %f (ms)\n",
				((cpu_worker_arg_t *) user_data)->id, run_time * 1.0e-6);
	}

	cutStartThread(&cpu_postprocess, user_data);
}*/

/**
 * Check whether all given device indexes are valid devices for the given context
 * @param context - the OpenCL context to look at
 * @param nDevices - total count of devices (size of deviceIds)
 * @param deviceIdxs - the device indexes
 * @param devices - list of devices to
 * @param log - whether logging had been previously enabled and is desired
 * @return -1 on failure, CL_SUCCESS otherwise
 */
int getDevices(cl_context context, int nDevices, unsigned int deviceIdxs[],
		cl_device_id* devices, bool log = false) {
	for (int iDevId = 0; iDevId < nDevices; iDevId++) {
		cl_device_id device = oclGetDev(context, deviceIdxs[iDevId]);
		if (device == (cl_device_id) -1) {
			if (log) {
				shrLog(" Device %d does not exist!\n", deviceIdxs[iDevId]);
			}
			return -1;
		}
		devices[iDevId] = device;
	}
	return CL_SUCCESS;
}

