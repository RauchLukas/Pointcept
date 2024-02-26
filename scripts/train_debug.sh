#!/bin/sh

# cd's in the parent folder of "scripts" -> effectivly the Pointcept folder
cd $(dirname $(dirname "$0")) || exit

echo "Root dir: $(pwd)"
ROOT_DIR=$(pwd)

# Default Settings
PYTHON=python

TRAIN_CODE=train.py
DEBUG_CODE=debug.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
GPU=None


while getopts "p:d:c:n:w:g:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
  echo "No GPU selected, but GPUs available: $NUM_GPU"
fi

echo "\n ============> CONFIG <============"
echo "PYTHON INTERPRETER:\t" $PYTHON
echo "DATASET NAME:\t\t" $DATASET
echo "CONFIG FILE NAME:\t" $CONFIG
echo "EXPERIMENT FOLDER:\t" $EXP_NAME
echo "RESUME TRAINING\t\t" $RESUME
echo "WEIGHTS PATH:\t\t" $WEIGHT
echo "GPUS:\t\t\t" $GPU
echo " =============> PATHS <=============="

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py

echo "EXPORT DIR:\t" $EXP_DIR
echo "MODEL DIR:\t" $MODEL_DIR
echo "CODE DIR:\t" $CODE_DIR
echo "EXPORT DIR:\t" $EXP_DIR
echo "CONFIG DIR:\t" $CONFIG_DIR

echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  echo "makedir: " $MODEL_DIR
  echo "makedir: " $CODE_DIR
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  echo "Copy folders /scrits /tools /pointcept to :" $CODE_DIR
  cp -r scripts tools pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

echo "[DEBUG] config dir :" $CONFIG_DIR

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR"
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi