################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/ClState.cpp \
../src/multithreading.cpp \
../src/sat.cpp 

CL_SRCS += \
../src/gpufilter.cl \
../src/sat.cl 

OBJS += \
./src/ClState.o \
./src/gpufilter.o \
./src/multithreading.o \
./src/sat.o 

CL_DEPS += \
./src/gpufilter.d \
./src/sat.d 

CPP_DEPS += \
./src/ClState.d \
./src/multithreading.d \
./src/sat.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D__GXX_EXPERIMENTAL_CXX0X__ -I"/home/algomorph/Factory/clsat/common/inc" -I/opt/intel/opencl-1.2-3.2.1.16712/include -I"/home/algomorph/Factory/clsat/include" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cl
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D__GXX_EXPERIMENTAL_CXX0X__ -I"/home/algomorph/Factory/clsat/common/inc" -I/opt/intel/opencl-1.2-3.2.1.16712/include -I"/home/algomorph/Factory/clsat/include" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


