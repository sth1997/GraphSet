ADD_EXECUTABLE(baseline_test baseline_test.cpp)
TARGET_LINK_LIBRARIES(baseline_test graph_mining)

#ADD_EXECUTABLE(sigmod2020_test sigmod2020_test.cpp)
#TARGET_LINK_LIBRARIES(sigmod2020_test graph_mining)

#ADD_EXECUTABLE(triangle_test triangle_test.cpp)
#TARGET_LINK_LIBRARIES(triangle_test graph_mining)

#ADD_EXECUTABLE(in_exclusion_optimize_test in_exclusion_optimize_test.cpp)
#TARGET_LINK_LIBRARIES(in_exclusion_optimize_test graph_mining)

#ADD_EXECUTABLE(all_schedule_test all_schedule_test.cpp)
#TARGET_LINK_LIBRARIES(all_schedule_test graph_mining)

#ADD_EXECUTABLE(performance_test performance_test.cpp)
#TARGET_LINK_LIBRARIES(performance_test graph_mining)

#ADD_EXECUTABLE(schedules_test schedules_test.cpp)
#TARGET_LINK_LIBRARIES(schedules_test graph_mining)

#ADD_EXECUTABLE(schedules_our_generator_test schedules_our_generator_test.cpp)
#TARGET_LINK_LIBRARIES(schedules_our_generator_test graph_mining)

ADD_EXECUTABLE(mc_ref mc_ref.cc)
TARGET_LINK_LIBRARIES(mc_ref graph_mining)
