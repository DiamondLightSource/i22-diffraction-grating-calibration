#!/bin/bash 
for dist in 5.8 7.2 9 12; do 
for angle in 10 25 50 75; do 
for seed in 0 1; do 
dir="output/d${dist}_a${angle}_s${seed}" 
nxs=$(ls "$dir"/*.nxs) 
out=$(python -m gratingcalibration --file "$nxs" --output-path "$dir/processing" 2>&1) 
echo $out > $dir/log.out 2>&1 
rot=$(echo "$out" | grep "Pattern rotation") 
det=$(echo "$out" | grep "Detector located") 
if [ -z "$det" ]; then 
echo "$dir: FAILED" 
else 
echo "$dir (true dist=$dist true angle=$angle): $rot | $det" 
fi 
done 
done 
done 
