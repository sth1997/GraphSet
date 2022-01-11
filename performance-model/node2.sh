#!/bin/bash
set -x

../build/bin/all_schedule_test ~/dataset/patents.g 7 0111111101111111011101110100111100011100001100000 > ./patents_p5.log_$(date -Iseconds)
