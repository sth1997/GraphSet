#!/bin/bash
source /opt/spack/share/spack/setup-env.sh
spack load cuda@11.3.1
spack load cmake@3.21.4%gcc@10.2.1
spack load --first openmpi@4.1.1%gcc@10.2.1