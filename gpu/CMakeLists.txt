ADD_EXECUTABLE(gpu_graph gpu_graph.cu)
SET_PROPERTY(TARGET gpu_graph PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_graph graph_mining)

ADD_EXECUTABLE(gpu_clique gpu_clique.cu)
SET_PROPERTY(TARGET gpu_clique PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_clique graph_mining)

ADD_EXECUTABLE(gpu_graph_mpi gpu_graph_mpi.cu)
SET_PROPERTY(TARGET gpu_graph_mpi PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_graph_mpi graph_mining mpi gomp)

ADD_EXECUTABLE(gpu_kclique gpu_kclique.cu)
SET_PROPERTY(TARGET gpu_kclique PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_kclique graph_mining)

ADD_EXECUTABLE(gpu_mc gpu_mc.cu)
SET_PROPERTY(TARGET gpu_mc PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_mc graph_mining)

ADD_EXECUTABLE(gpu_graph_unimem gpu_graph_unimem.cu)
SET_PROPERTY(TARGET gpu_graph_unimem PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_graph_unimem graph_mining mpi gomp)

ADD_EXECUTABLE(fsm_mixed fsm_mixed.cu)
SET_PROPERTY(TARGET fsm_mixed PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(fsm_mixed graph_mining)

ADD_EXECUTABLE(gpu_new_str gpu_new_str.cu)
SET_PROPERTY(TARGET gpu_new_str PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_new_str graph_mining mpi gomp)

ADD_EXECUTABLE(gpu_fsm gpu_fsm.cu)
SET_PROPERTY(TARGET gpu_fsm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_fsm graph_mining)

ADD_EXECUTABLE(gpu_new_fsm gpu_new_fsm.cu)
SET_PROPERTY(TARGET gpu_new_fsm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_new_fsm graph_mining)

ADD_EXECUTABLE(gpu_fsm_omp gpu_fsm_omp.cu)
SET_PROPERTY(TARGET gpu_fsm_omp PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_fsm_omp graph_mining)

ADD_EXECUTABLE(gpu_graph_static_task gpu_graph_static_task.cu)
SET_PROPERTY(TARGET gpu_graph_static_task PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_graph_static_task graph_mining mpi gomp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu_new_str OpenMP::OpenMP_CXX)
    target_link_libraries(gpu_fsm_omp OpenMP::OpenMP_CXX)
    target_link_libraries(gpu_kclique OpenMP::OpenMP_CXX)
    target_link_libraries(gpu_graph_mpi OpenMP::OpenMP_CXX)
    target_link_libraries(gpu_graph_unimem OpenMP::OpenMP_CXX)
    target_link_libraries(gpu_new_fsm OpenMP::OpenMP_CXX)
    target_link_libraries(fsm_mixed OpenMP::OpenMP_CXX)
endif()

ADD_EXECUTABLE(mc3 mc3.cu)
SET_PROPERTY(TARGET mc3 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(mc3 graph_mining)
