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

cat before_inj.cu > pt_p1.cu
cat pt_p1_inject.cu >> pt_p1.cu
cat after_inj.cu >> pt_p1.cu

cat before_inj.cu > pt_p2.cu
cat pt_p2_inject.cu >> pt_p2.cu
cat after_inj.cu >> pt_p2.cu

cat before_inj.cu > pt_p3.cu
cat pt_p3_inject.cu >> pt_p3.cu
cat after_inj.cu >> pt_p3.cu

cat before_inj.cu > pt_p4.cu
cat pt_p4_inject.cu >> pt_p4.cu
cat after_inj.cu >> pt_p4.cu

cat before_inj.cu > pt_p5.cu
cat pt_p5_inject.cu >> pt_p5.cu
cat after_inj.cu >> pt_p5.cu

cat before_inj.cu > pt_p6.cu
cat pt_p6_inject.cu >> pt_p6.cu
cat after_inj.cu >> pt_p6.cu

cat before_inj.cu > mc_p1.cu
cat mc_p1_inject.cu >> mc_p1.cu
cat after_inj.cu >> mc_p1.cu

cat before_inj.cu > mc_p2.cu
cat mc_p2_inject.cu >> mc_p2.cu
cat after_inj.cu >> mc_p2.cu

cat before_inj.cu > mc_p3.cu
cat mc_p3_inject.cu >> mc_p3.cu
cat after_inj.cu >> mc_p3.cu

cat before_inj.cu > mc_p4.cu
cat mc_p4_inject.cu >> mc_p4.cu
cat after_inj.cu >> mc_p4.cu

cat before_inj.cu > mc_p5.cu
cat mc_p5_inject.cu >> mc_p5.cu
cat after_inj.cu >> mc_p5.cu

cat before_inj.cu > mc_p6.cu
cat mc_p6_inject.cu >> mc_p6.cu
cat after_inj.cu >> mc_p6.cu

cat before_inj.cu > wv_p1.cu
cat wv_p1_inject.cu >> wv_p1.cu
cat after_inj.cu >> wv_p1.cu

cat before_inj.cu > wv_p2.cu
cat wv_p2_inject.cu >> wv_p2.cu
cat after_inj.cu >> wv_p2.cu

cat before_inj.cu > wv_p3.cu
cat wv_p3_inject.cu >> wv_p3.cu
cat after_inj.cu >> wv_p3.cu

cat before_inj.cu > wv_p4.cu
cat wv_p4_inject.cu >> wv_p4.cu
cat after_inj.cu >> wv_p4.cu

cat before_inj.cu > wv_p5.cu
cat wv_p5_inject.cu >> wv_p5.cu
cat after_inj.cu >> wv_p5.cu

cat before_inj.cu > wv_p6.cu
cat wv_p6_inject.cu >> wv_p6.cu
cat after_inj.cu >> wv_p6.cu

cat before_inj.cu > lj_p1.cu
cat lj_p1_inject.cu >> lj_p1.cu
cat after_inj.cu >> lj_p1.cu

cat before_inj.cu > lj_p2.cu
cat lj_p2_inject.cu >> lj_p2.cu
cat after_inj.cu >> lj_p2.cu

cat before_inj.cu > lj_p3.cu
cat lj_p3_inject.cu >> lj_p3.cu
cat after_inj.cu >> lj_p3.cu

cat before_inj.cu > lj_p4.cu
cat lj_p4_inject.cu >> lj_p4.cu
cat after_inj.cu >> lj_p4.cu

cat before_inj.cu > lj_p5.cu
cat lj_p5_inject.cu >> lj_p5.cu
cat after_inj.cu >> lj_p5.cu

cat before_inj.cu > lj_p6.cu
cat lj_p6_inject.cu >> lj_p6.cu
cat after_inj.cu >> lj_p6.cu

cat before_inj.cu > or_p1.cu
cat or_p1_inject.cu >> or_p1.cu
cat after_inj.cu >> or_p1.cu

cat before_inj.cu > or_p2.cu
cat or_p2_inject.cu >> or_p2.cu
cat after_inj.cu >> or_p2.cu

cat before_inj.cu > or_p3.cu
cat or_p3_inject.cu >> or_p3.cu
cat after_inj.cu >> or_p3.cu

cat before_inj.cu > or_p4.cu
cat or_p4_inject.cu >> or_p4.cu
cat after_inj.cu >> or_p4.cu

cat before_inj.cu > or_p5.cu
cat or_p5_inject.cu >> or_p5.cu
cat after_inj.cu >> or_p5.cu

cat before_inj.cu > or_p6.cu
cat or_p6_inject.cu >> or_p6.cu
cat after_inj.cu >> or_p6.cu

