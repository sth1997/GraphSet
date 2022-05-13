ADD_EXECUTABLE(gpu_house gpu_house.cu)
SET_PROPERTY(TARGET gpu_house PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_house graph_mining)

ADD_EXECUTABLE(gpu_graph gpu_graph.cu)
SET_PROPERTY(TARGET gpu_graph PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_graph graph_mining)

ADD_EXECUTABLE(gpu_clique gpu_clique.cu)
SET_PROPERTY(TARGET gpu_clique PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_clique graph_mining)

ADD_EXECUTABLE(gpu_graph_with_timer gpu_graph_with_timer.cu)
SET_PROPERTY(TARGET gpu_graph_with_timer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_graph_with_timer graph_mining)

ADD_EXECUTABLE(gpu_house_with_timer gpu_house_with_timer.cu)
SET_PROPERTY(TARGET gpu_house_with_timer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_house_with_timer graph_mining)

ADD_EXECUTABLE(gpu_fsm gpu_fsm.cu)
SET_PROPERTY(TARGET gpu_fsm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_fsm graph_mining)


ADD_EXECUTABLE(gpu_fsm_omp gpu_fsm_omp.cu)
SET_PROPERTY(TARGET gpu_fsm_omp PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_fsm_omp graph_mining)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu_fsm_omp OpenMP::OpenMP_CXX)
endif()