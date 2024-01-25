#!/bin/bash

CONDA_SCRIPT=$CONDA_DIR/etc/profile.d/conda.sh
CONDA_ENV=clrnet
CONFIG_FILE=configs/clrnet/clr_dla34_tusimple.py
CONVERT_SCRIPT=convert.py
DETECT_SCRIPT=tools/detect.py
JSON_FILE=pred.json
MODEL_PATH=/home/ssd/liyufan/clrnet/best.pth

function main() {
    python $DETECT_SCRIPT \
        $CONFIG_FILE \
        --img $1 \
        --load_from $MODEL_PATH \
        --savedir $1
    python $CONVERT_SCRIPT \
        --save_dir $1 \
        --json_file $JSON_FILE
    rm $1/$JSON_FILE
}

if [ $# -ne 1 ]; then
    echo "Usage: $0 <img | img_dir>"
    exit 1
fi

if [ ! -d $1 ]; then
    echo "Error: $1 is not a dir"
    exit 1
fi

. $CONDA_SCRIPT && conda activate $CONDA_ENV
main $1
