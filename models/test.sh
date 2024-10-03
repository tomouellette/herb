#!/bin/bash

for model in dino mae mbt; do
    echo "[INFO | test] Testing $model model."
    python3 -m models.$model \
        --input data/test \
        --output test/$model \
        --epochs 2 \
        --image_size 64 \
        --batch_size 12 \
        --print_fraction 0 

    if [ -f test/${model}/*/logger.pt ]; then
        echo "[INFO | test] $model test passed."
        rm -rf test/mbt
    else
        echo "[ERROR | test] $model test failed."
        exit 1
    fi
done

echo ""
echo "[INFO | test] All tests passed."

rm -rf test
