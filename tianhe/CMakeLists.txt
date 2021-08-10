ADD_EXECUTABLE(baseline_test baseline_test.cpp)
TARGET_LINK_LIBRARIES(baseline_test graph_mining)

ADD_EXECUTABLE(motifs_test motifs_test.cpp)
TARGET_LINK_LIBRARIES(motifs_test graph_mining)

