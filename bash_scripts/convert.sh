#!/bin/bash
singularity exec /home/hujin/jin/singularity_containers/base_latest.sif jupyter nbconvert --to script $1
