#!/bin/bash
set -x

../build/bin/baseline_test ~/dataset/orkut.g 4 0110100110010110 > ./orkut.g_motif_p7.log_$(date -Iseconds)
