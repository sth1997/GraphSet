#pragma once

int print_device_information() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf("Device Information:");
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d, name: %s\n", i, prop.name);
        if (i == 0) {
            printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
            printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %.2lf\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        }
    }
    return nDevices;
}