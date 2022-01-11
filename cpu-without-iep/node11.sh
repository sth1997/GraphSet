#!/bin/bash
set -x

../build/bin/baseline_test LiveJournal ~/dataset/livejournal_input 5 0111010011100011100001100 > ./livejournal_p1.log_$(date -Iseconds)
../build/bin/baseline_test Orkut ~/dataset/orkut_input 5 0111010011100011100001100 > ./orkut_p1.log_$(date -Iseconds)
../build/bin/baseline_test Orkut ~/dataset/orkut_input 6 011011101110110101011000110000101000 > ./orkut_p2.log_$(date -Iseconds)
../build/bin/baseline_test Orkut ~/dataset/orkut_input 6 011111101000110111101010101101101010 > ./orkut_p3.log_$(date -Iseconds)
