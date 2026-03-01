#!/bin/bash
# Wrapper to run scene2 in clavic env
eval "$(conda shell.bash hook)"
conda activate clavic
cd /home/siddharth/ssd_data/clavic
python -u main_scene2.py 2>&1 | tee scene2_log.txt
