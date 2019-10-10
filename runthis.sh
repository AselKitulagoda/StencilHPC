#!/bin/bash
make -B
sbatch stencil.job
sleep 5
less stencil.out 
