#!/bin/bash

srun ../gpu_triangle ~zms/patents_input > patents
srun ../gpu_triangle ~zms/orkut_input > orkut
srun ../gpu_triangle ~zms/livejournal_input > livejournal
srun ../gpu_triangle ~zms/wiki-vote_input > wiki-vote
srun ../gpu_triangle ~zms/mico_input > mico
