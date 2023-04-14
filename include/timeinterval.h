#pragma once
#include <sys/time.h>
#include <chrono>

class TimeInterval{
public:
    TimeInterval(){
        check();
    }

    void check(){
        gettimeofday(&tp, NULL);
    }

    double print(const char* title){
        struct timeval tp_end, tp_res;
        gettimeofday(&tp_end, NULL);
        timersub(&tp_end, &tp, &tp_res);
        printf("%s: %ld.%06ld s.\n", title, tp_res.tv_sec, tp_res.tv_usec);
        return tp_res.tv_sec + tp_res.tv_usec / 1e6;
    }
private:
    struct timeval tp;
};
