#!/bin/bash

srun ../nonCDP_house Patents ~zms/patents_input > patents
srun ../nonCDP_house Orkut ~zms/orkut_input > orkut
srun ../nonCDP_house LiveJournal ~zms/livejournal_input > livejournal
srun ../nonCDP_house Wiki-Vote ~zms/wiki-vote_input > wiki-vote
srun ../nonCDP_house MiCo ~zms/mico_input > mico
