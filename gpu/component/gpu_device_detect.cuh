#pragma once
#include <iostream>

int get_device_information(bool print = false) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if(print) std::cerr << "Device Information:" << std::endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if(print) std::cerr << "Device " << i << ", name: " << prop.name << std::endl;
        if (i == 0) {
            if(print) std::cerr << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
            if(print) std::cerr << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
            if(print) std::cerr.precision(2);
            if(print) std::cerr << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
        }
    }
    return nDevices;
}