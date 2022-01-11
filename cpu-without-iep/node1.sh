#!/bin/bash
set -x

../build/bin/baseline_test MiCo ~/dataset/mico_input 7 0111111101111111011101110100111100011100001100000 > ./mico_p5.log_$(date -Iseconds)
