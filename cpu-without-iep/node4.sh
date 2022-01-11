#!/bin/bash
set -x

../build/bin/baseline_test ~/dataset/livejournal.g 6 011111101000110111101010101101101010 > ./livejournal_p3.log_$(date -Iseconds)
