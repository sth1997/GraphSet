#!/bin/bash
<< comment
for i in {1..5}; do
    # echo $i
    echo "[baseline orkut]" >> output.txt
    bin/baseline_test Orkut ~zms/orkut_input | tee -a output.txt
    echo "[gpu_graph orkut]" >> output.txt
    ../../GraphMining-GPU/build/bin/gpu_graph Orkut ~zms/orkut_input | tee -a output.txt
    echo "[gpu_graph* orkut]" >> output.txt
    bin/gpu_graph Orkut ~zms/orkut_input | tee -a output.txt
    echo "[gpu_house orkut]" >> output.txt
    bin/gpu_house Orkut ~zms/orkut_input | tee -a output.txt
done
comment

for i in {1..4}; do
    echo "[baseline livejournal]" >> output2.txt
    bin/baseline_test LiveJournal ~zms/livejournal_input | tee -a output2.txt
    echo "[gpu_graph livejournal]" >> output2.txt
    ../../GraphMining-GPU/build/bin/gpu_graph LiveJournal ~zms/livejournal_input | tee -a output2.txt
    echo "[gpu_graph* livejournal]" >> output2.txt
    bin/gpu_graph LiveJournal ~zms/livejournal_input | tee -a output2.txt
    echo "[gpu_house livejournal]" >> output2.txt
    bin/gpu_house LiveJournal ~zms/livejournal_input | tee -a output2.txt
done