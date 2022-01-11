#!/bin/bash
set -x

../build/bin/baseline_test ~/dataset/livejournal.g 6 011011101110110101011000110000101000 > ./livejournal_p2.log_$(date -Iseconds)
