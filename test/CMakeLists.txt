enable_testing()

set(GTEST_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/modules/googletest/googletest/include")
set(GTEST_LIB gtest_main gtest)
include_directories(${GTEST_INCLUDE_DIR})

include(GoogleTest)
MESSAGE("GTEST FOUND")

ADD_EXECUTABLE(pattern_matching_test pattern_matching_test.cpp)
TARGET_LINK_LIBRARIES(pattern_matching_test
                    ${GTEST_LIB}
                    graph_mining
                    pthread
                    gomp)
gtest_discover_tests(pattern_matching_test)

ADD_EXECUTABLE(gpu_sample_test gpu_sample_test.cu)
SET_PROPERTY(TARGET gpu_sample_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_sample_test
                    ${GTEST_LIB}
                    graph_mining
                    pthread
                    gomp)
  
gtest_discover_tests(gpu_sample_test)


ADD_EXECUTABLE(gpu_pattern_matching_test gpu_pattern_matching_test.cu)
SET_PROPERTY(TARGET gpu_pattern_matching_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(gpu_pattern_matching_test
                    ${GTEST_LIB}
                    graph_mining
                    pthread
                    gomp)
  
gtest_discover_tests(gpu_pattern_matching_test)

# ADD_EXECUTABLE(performance_modeling_test performance_modeling_test.cpp)
# TARGET_LINK_LIBRARIES(performance_modeling_test
#                     ${GTEST_LIB}
#                     graph_mining
#                     pthread
#                     gomp)
# gtest_discover_tests(performance_modeling_test)

# ADD_EXECUTABLE(restricts_generation_test restricts_generation_test.cpp)
# TARGET_LINK_LIBRARIES(restricts_generation_test
#                     ${GTEST_LIB}
#                     graph_mining
#                     pthread
#                     gomp)
# gtest_discover_tests(restricts_generation_test)

# ADD_EXECUTABLE(schedule_compare_test schedule_compare_test.cpp)
# TARGET_LINK_LIBRARIES(schedule_compare_test
#                     ${GTEST_LIB}
#                     graph_mining
#                     pthread
#                     gomp)
# gtest_discover_tests(schedule_compare_test)

# ADD_EXECUTABLE(restricts_compare_test restricts_compare_test.cpp)
# TARGET_LINK_LIBRARIES(restricts_compare_test
#                     ${GTEST_LIB}
#                     graph_mining
#                     pthread
#                     gomp)
# gtest_discover_tests(restricts_compare_test)

# ADD_EXECUTABLE(performance_compare_test performance_compare_test.cpp)
# TARGET_LINK_LIBRARIES(performance_compare_test
#                     ${GTEST_LIB}
#                     graph_mining
#                     pthread
#                     gomp)
# gtest_discover_tests(performance_compare_test)

