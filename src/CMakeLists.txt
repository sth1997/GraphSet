SET(GraphMiningSrc
graph.cpp
labeled_graph.cpp
dataloader.cpp
pattern.cpp
prefix.cpp
#schedule.cpp
schedule_IEP.cpp
vertex_set.cpp
motif_generator.cpp
graphmpi.cpp
common.cpp
disjoint_set_union.cpp
# set_operation.cpp
)

ADD_LIBRARY(graph_mining SHARED ${GraphMiningSrc}) 

ADD_EXECUTABLE(fsm_test fsm_test.cpp)
TARGET_LINK_LIBRARIES(fsm_test graph_mining)

ADD_EXECUTABLE(clique_test clique_test.cpp)
TARGET_LINK_LIBRARIES(clique_test graph_mining)

ADD_EXECUTABLE(motif_counting_test motif_counting_test.cpp)
TARGET_LINK_LIBRARIES(motif_counting_test graph_mining)

ADD_EXECUTABLE(pm_test pm_test.cpp)
TARGET_LINK_LIBRARIES(pm_test graph_mining)

#ADD_EXECUTABLE(in_exclusion_performance_test in_exclusion_performance_test.cpp)
#TARGET_LINK_LIBRARIES(in_exclusion_performance_test graph_mining)

#ADD_EXECUTABLE(performance_compare_test performance_compare_test.cpp)
#TARGET_LINK_LIBRARIES(performance_compare_test graph_mining)

#ADD_EXECUTABLE(schedule_printer schedule_printer.cpp)
#TARGET_LINK_LIBRARIES(schedule_printer graph_mining)

#ADD_EXECUTABLE(restricts_printer restricts_printer.cpp)
#TARGET_LINK_LIBRARIES(restricts_printer graph_mining)

#ADD_EXECUTABLE(run_brute_force_tc run_brute_force_tc.cpp)
#TARGET_LINK_LIBRARIES(run_brute_force_tc graph_mining)

#ADD_EXECUTABLE(run_general_tc run_general_tc.cpp)
#TARGET_LINK_LIBRARIES(run_general_tc graph_mining)
