#include <gtest/gtest.h>
#include <../include/graph.h>
#include <../include/dataloader.h>
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "../gpu/gpu_test.cuh"

#include <iostream>
#include <string>
#include <algorithm>

TEST(gpu_sample_test,gpu_sample_none_test){
    int n = 100000;
    int *a = new int[n];
    int *b = new int[n];
    for(int i = 0; i < n; i++) {
        a[i] = rand() % (0x3f3f3f3f);
        b[i] = rand() % (0x3f3f3f3f);
    }
    test(n, a, b);
    for(int i = 0; i < n; i++){
        ASSERT_EQ(a[i], b[i]);
    }
}