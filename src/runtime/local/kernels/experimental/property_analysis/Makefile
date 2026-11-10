# Build the thesis kernels explicitly; they are not part of libAllKernels or myKernels.
CXX = g++
ROOT := ../../../../../..
CPPFLAGS += -I$(ROOT)/src -I$(ROOT)/build/src -I$(ROOT)/thirdparty/installed/include -DSPDLOG_FMT_EXTERNAL
CXXFLAGS += -std=c++20 -O3 -fPIC -mavx2

libPropertyKernels.so: PropertyKernels.o
	$(CXX) -shared $< -o $@

PropertyKernels.o: PropertyKernels.cpp PiggybackAnalysis.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RM) PropertyKernels.o libPropertyKernels.so
