#!/bin/bash

source activate AdvFLYP

cd /leonardo_work/IscrC_AdvFLYP/AdvFLYP    # Specify your path to this project folder

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Training Setup
ROOT=/leonardo_work/IscrC_AdvFLYP/data # Specify your path to the /data folder
SEED=42
NAME=AdvFLYP_full
DATASET=smallLAION
N_TRAIN_DATA=1000000
MODEL_DIR=./save_AdvFLYP
LAMBDA_FEAT=1
LAMBDA_LOGIT=1
REG_METHOD=(logit feat)
EVAL_SET=tinyImageNet
BATCH_SIZE=256
EPOCHS=100
LR=1e-4
TRAIN_ATTACK_TYPE=pgd
TRAIN_EPS=1
TRAIN_NUMSTEPS=2
TRAIN_STEPSIZE=1
TEST_ATTACK_TYPE=pgd
TEST_EPS=1
TEST_NUMSTEPS=10
TEST_STEPSIZE=1

python -m code.main \
    --root $ROOT \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LR \
    --dataset $DATASET \
    --n_data $N_TRAIN_DATA \
    --lambda_feat $LAMBDA_FEAT \
    --lambda_logit $LAMBDA_LOGIT \
    --reg_level "${REG_METHOD[@]}" \
    --model_dir $MODEL_DIR \
    --name $NAME \
    --train_attack_type $TRAIN_ATTACK_TYPE \
    --train_eps $TRAIN_EPS \
    --train_numsteps $TRAIN_NUMSTEPS \
    --train_stepsize $TRAIN_STEPSIZE \
    --eval_set $EVAL_SET \
    --test_attack_type $TEST_ATTACK_TYPE \
    --test_eps $TEST_EPS \
    --test_numsteps $TEST_NUMSTEPS \
    --test_stepsize $TEST_STEPSIZE
