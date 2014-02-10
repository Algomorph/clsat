################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../common/src/cmd_arg_reader.cpp \
../common/src/oclUtils.cpp \
../common/src/shrUtils.cpp 

OBJS += \
./common/src/cmd_arg_reader.o \
./common/src/oclUtils.o \
./common/src/shrUtils.o 

CPP_DEPS += \
./common/src/cmd_arg_reader.d \
./common/src/oclUtils.d \
./common/src/shrUtils.d 


# Each subdirectory must supply rules for building sources it contributes
common/src/%.o: ../common/src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D__GXX_EXPERIMENTAL_CXX0X__ -I"/home/algomorph/Factory/clsat/common/inc" -I/opt/intel/opencl-1.2-3.2.1.16712/include -I"/home/algomorph/Factory/clsat/include" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


