#!/bin/bash

source activate AdvFLYP

cd /leonardo_work/IscrC_AdvFLYP/AdvFLYP

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Test Setup
TEST_MODEL_PATH=CHECKPOINT.pth.tar      # Specify path to the model checkpoints to be evaluated
ROOT=/leonardo_work/IscrC_AdvFLYP/data  # Specify path to /data
TEST_SET=(cifar10 cifar100 STL10 Caltech101 Caltech256 oxfordpet flowers102 Food101 StanfordCars SUN397 Country211 fgvc_aircraft EuroSAT dtd)
TEST_ATTACK_TYPE=pgd
TEST_EPS=1
TEST_NUMSTEPS=10
TEST_STEPSIZE=1
BATCH_SIZE=256
python -m code.main \
    --evaluate \
    --resume $TEST_MODEL_PATH \
    --test_set "${TEST_SET[@]}" \
    --root $ROOT \
    --batch_size $BATCH_SIZE \
    --test_attack_type $TEST_ATTACK_TYPE \
    --test_eps $TEST_EPS \
    --test_numsteps $TEST_NUMSTEPS \
    --test_stepsize $TEST_STEPSIZE
