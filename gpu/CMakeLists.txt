ADD_EXECUTABLE(gpu_house gpu_house.cu)
SET_PROPERTY(TARGET gpu_house PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_house graph_mining gomp)

ADD_EXECUTABLE(gpu_triangle gpu_triangle.cu)
SET_PROPERTY(TARGET gpu_triangle PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_triangle graph_mining gomp)

ADD_EXECUTABLE(nonCDP_house nonCDP_house.cu)
SET_PROPERTY(TARGET nonCDP_house PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(nonCDP_house graph_mining gomp)

ADD_EXECUTABLE(gpu_test gpu_test.cu)
TARGET_LINK_LIBRARIES(gpu_test graph_mining gomp)

ADD_EXECUTABLE(check check.cu)
TARGET_LINK_LIBRARIES(check graph_mining gomp)
