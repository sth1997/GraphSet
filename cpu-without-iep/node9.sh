#!/bin/bash
set -x

../build/bin/baseline_test Orkut ~/dataset/orkut_input 7 0111111101111111011101110100111100011100001100000 > ./orkut_p5.log_$(date -Iseconds)
