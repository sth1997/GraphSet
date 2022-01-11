#!/bin/bash
set -x

../build/bin/baseline_test ~/dataset/livejournal.g 7 0111111101111111011001110100111100011000001100000 > ./livejournal_p6.log_$(date -Iseconds)
../build/bin/baseline_test ~/dataset/orkut.g 5 0111010011100011100001100 > ./orkut_p1.log_$(date -Iseconds)
../build/bin/baseline_test ~/dataset/orkut.g 6 011011101110110101011000110000101000 > ./orkut_p2.log_$(date -Iseconds)
../build/bin/baseline_test ~/dataset/orkut.g 6 011111101000110111101010101101101010 > ./orkut_p3.log_$(date -Iseconds)
