#!/bin/bash
set -x

../build/bin/all_schedule_test ~/dataset/patents.g 5 0111010011100011100001100 > ./patents_p1.log_$(date -Iseconds)
../build/bin/all_schedule_test ~/dataset/patents.g 6 011011101110110101011000110000101000 > ./patents_p2.log_$(date -Iseconds)
../build/bin/all_schedule_test ~/dataset/patents.g 6 011111101000110111101010101101101010 > ./patents_p3.log_$(date -Iseconds)
../build/bin/all_schedule_test ~/dataset/patents.g 6 011110101101110000110000100001010010 > ./patents_p4.log_$(date -Iseconds)
