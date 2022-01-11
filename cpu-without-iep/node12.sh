#!/bin/bash
set -x

../build/bin/baseline_test Wiki-Vote ~/dataset/wiki-vote_input 5 0111010011100011100001100 > ./wiki-vote_p1.log_$(date -Iseconds)
../build/bin/baseline_test Wiki-Vote ~/dataset/wiki-vote_input 6 011011101110110101011000110000101000 > ./wiki-vote_p2.log_$(date -Iseconds)
../build/bin/baseline_test Wiki-Vote ~/dataset/wiki-vote_input 6 011111101000110111101010101101101010 > ./wiki-vote_p3.log_$(date -Iseconds)
../build/bin/baseline_test Wiki-Vote ~/dataset/wiki-vote_input 6 011110101101110000110000100001010010 > ./wiki-vote_p4.log_$(date -Iseconds)
../build/bin/baseline_test Wiki-Vote ~/dataset/wiki-vote_input 7 0111111101111111011101110100111100011100001100000 > ./wiki-vote_p5.log_$(date -Iseconds)
../build/bin/baseline_test Wiki-Vote ~/dataset/wiki-vote_input 7 0111111101111111011001110100111100011000001100000 > ./wiki-vote_p6.log_$(date -Iseconds)
../build/bin/baseline_test Patents ~/dataset/patents_input 5 0111010011100011100001100 > ./patents_p1.log_$(date -Iseconds)
../build/bin/baseline_test Patents ~/dataset/patents_input 6 011011101110110101011000110000101000 > ./patents_p2.log_$(date -Iseconds)
../build/bin/baseline_test Patents ~/dataset/patents_input 6 011111101000110111101010101101101010 > ./patents_p3.log_$(date -Iseconds)
../build/bin/baseline_test Patents ~/dataset/patents_input 6 011110101101110000110000100001010010 > ./patents_p4.log_$(date -Iseconds)
../build/bin/baseline_test Patents ~/dataset/patents_input 7 0111111101111111011101110100111100011100001100000 > ./patents_p5.log_$(date -Iseconds)
../build/bin/baseline_test Patents ~/dataset/patents_input 7 0111111101111111011001110100111100011000001100000 > ./patents_p6.log_$(date -Iseconds)
../build/bin/baseline_test MiCo ~/dataset/mico_input 5 0111010011100011100001100 > ./mico_p1.log_$(date -Iseconds)
../build/bin/baseline_test MiCo ~/dataset/mico_input 6 011011101110110101011000110000101000 > ./mico_p2.log_$(date -Iseconds)
../build/bin/baseline_test MiCo ~/dataset/mico_input 6 011111101000110111101010101101101010 > ./mico_p3.log_$(date -Iseconds)
../build/bin/baseline_test MiCo ~/dataset/mico_input 6 011110101101110000110000100001010010 > ./mico_p4.log_$(date -Iseconds)
