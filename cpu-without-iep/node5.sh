#!/bin/bash
set -x

../build/bin/baseline_test ~/dataset/livejournal.g 6 011110101101110000110000100001010010 > ./livejournal_p4.log_$(date -Iseconds)
