#!/bin/bash
set -x

../build/bin/baseline_test ~/dataset/livejournal.g 4 0110100110010110 > ./livejournal.g_motif_p7.log_$(date -Iseconds)
