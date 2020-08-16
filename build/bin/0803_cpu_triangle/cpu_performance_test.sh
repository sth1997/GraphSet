#!/bin/bash

srun ../triangle_test Patents ~zms/patents_input > patents &
srun ../triangle_test Orkut ~zms/orkut_input > orkut &
srun ../triangle_test LiveJournal ~zms/livejournal_input > livejournal &
srun ../triangle_test Wiki-Vote ~zms/wiki-vote_input > wiki-vote &
srun ../triangle_test MiCo ~zms/mico_input > mico &
