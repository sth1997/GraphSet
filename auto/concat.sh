#!/bin/bash
p1="5 0111010011100011100001100"
p2="6 011011101110110101011000110000101000"
p3="6 011111101000110111101010101101101010"
p4="6 011110101101110000110000100001010010"
p5="7 0111111101111111011101110100111100011100001100000"
p6="7 0111111101111111011001110100111100011000001100000"


pt="/home/hzx/data/patents.g"
mc="/home/hzx/data/mico.g"
wv="/home/hzx/data/wiki-vote.g"
lj="/home/hzx/data/livejournal.g"
or="/home/hzx/data/orkut.g"


for name in "pt" "mc" "lj" "or"
do
    for ((i=1; i<=9; i++))
    do
        cat before_inj.cu > ${name}_p${i}.cu
        cat ${name}_p${i}_inject.cu >> ${name}_p${i}.cu
        cat after_inj.cu >> ${name}_p${i}.cu
    done
done