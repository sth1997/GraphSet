#!/bin/bash
source /opt/spack/share/spack/setup-env.sh
spack load cuda@12.0
spack load cmake@3.26
spack load openmpi@4.1.5