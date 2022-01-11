#!/bin/bash
set -x

../build/bin/baseline_test Orkut ~/dataset/orkut_input 6 011110101101110000110000100001010010 > ./orkut_p4.log_$(date -Iseconds)
