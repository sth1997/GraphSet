#!/bin/bash
set -x

../build/bin/baseline_test Orkut ~/dataset/orkut_input 7 0111111101111111011001110100111100011000001100000 > ./orkut_p6.log_$(date -Iseconds)
