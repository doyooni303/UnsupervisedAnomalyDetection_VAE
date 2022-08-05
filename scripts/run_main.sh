#!/bin/bash
#-save_final_model True -group_name Final_Train -save_epochs False

PARMETER_DIR="VAE_THR_TH_02_LR_1e-2"
PROJECT='Dacon Unsupervised Anomaly Detection'
DEVICE='cuda:0'
DATA_PATH='data'
NOTE='VAE with Threshold:0.2 / Learningrate:1e-2'
EPOCHS=10
THRESHOLD=0.1
ENC_HIDDEN_DIM=32
LATENT_DIM=16
DEC_HIDDEN_DIM=32
SEED=42
OPTIMIZER='adam'
ACCUM_STEPS=1
LR=1e-2
MOMENTUM=0.9
BATCH_SIZE=16
VAL_BATCH_SIZE=512
SAVE_BEST='True'

python3 src/main.py\
    --parameter_dir "$PARMETER_DIR"\
    --project "$PROJECT"\
    --device $DEVICE\
    --data_path $DATA_PATH\
    --note "$NOTE"\
    --epochs $EPOCHS\
    --threshold $THRESHOLD\
    --enc_hidden_dim $ENC_HIDDEN_DIM\
    --latent_dim $LATENT_DIM\
    --dec_hidden_dim $DEC_HIDDEN_DIM\
    --seed $SEED\
    --optimizer $OPTIMIZER\
    --accum_steps $ACCUM_STEPS\
    --lr $LR\
    --momentum $MOMENTUM\
    --batch_size $BATCH_SIZE\
    --val_batch_size $VAL_BATCH_SIZE\
    --save_best $SAVE_BEST\