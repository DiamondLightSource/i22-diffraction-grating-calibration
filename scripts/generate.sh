#!/bin/bash
for dist in 5.8 7.2 9 12; do 
for angle in 10 25 50 75; do 
for seed in 0 1; do 
dir="output/d${dist}_a${angle}_s${seed}" 
mkdir -p "$dir" 
python generate_synthetic_data.py --distance $dist --pattern-angle $angle --seed $seed --output-path "$dir" > /dev/null 
done 
done 
done 
echo "done generating" 
ls scripts | grep '^d' | wc -l 
