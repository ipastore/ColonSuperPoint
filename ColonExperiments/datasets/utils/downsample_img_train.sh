#!/bin/bash

SRC_DIR="/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/3
3/images"
DST_DIR="/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/toy_33/images"

mkdir -p "$DST_DIR"

# List all images, sort naturally, get total count
mapfile -t FILES < <(ls "$SRC_DIR" | sort -V)
TOTAL=${#FILES[@]}
NUM_SELECT=100

# Compute stride (skip factor)
STEP=$(( TOTAL / NUM_SELECT ))

# Downsample and skip the first frame
COUNT=0
for (( i=STEP; i<TOTAL; i+=STEP )); do
    if (( COUNT >= NUM_SELECT )); then break; fi
    FILENAME="${FILES[$i]}"
    cp "$SRC_DIR/$FILENAME" "$DST_DIR/$FILENAME"
    ((COUNT++))
done

echo "Copied $COUNT images from $SRC_DIR to $DST_DIR"
