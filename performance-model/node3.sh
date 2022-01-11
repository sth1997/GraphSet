#!/bin/bash
set -x

../build/bin/all_schedule_test ~/dataset/patents.g 7 0111111101111111011001110100111100011000001100000 > ./patents_p6.log_$(date -Iseconds)
