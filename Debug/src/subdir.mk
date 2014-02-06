################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/ClState.cpp \
../src/multithreading.cpp \
../src/oclMultiThreads.cpp \
../src/sat.cpp 

CL_SRCS += \
../src/gpufilter.cl \
../src/sat.cl 

OBJS += \
./src/ClState.o \
./src/multithreading.o \
./src/oclMultiThreads.o \
./src/sat.o 

CPP_DEPS += \
./src/ClState.d \
./src/multithreading.d \
./src/oclMultiThreads.d \
./src/sat.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D__GXX_EXPERIMENTAL_CXX0X__ -I"/home/algomorph/Factory/clsat/common/inc" -I"/home/algomorph/Factory/clsat/include" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/gpufilter: ../src/gpufilter.cl
	@echo 'Building file: $<'
	@echo 'Invoking: Resource Custom Build Step'
	
	@echo 'Finished building: $<'
	@echo ' '

src/sat: ../src/sat.cl
	@echo 'Building file: $<'
	@echo 'Invoking: Resource Custom Build Step'
	
	@echo 'Finished building: $<'
	@echo ' '


