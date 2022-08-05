#!/bin/bash
#-save_final_model True -group_name Final_Train -save_epochs False

PARMETER_DIR='VAE_THR_1e-3'
PROJECT='Dacon Unsupervised Anomaly Detection'
DEVICE='cuda:0'
DATA_PATH='data'
NOTE='VAE with Threshold:1e-3'
EPOCHS=10
THRESHOLD=1e-3
ENC_HIDDEN_DIM=32
LATENT_DIM=16
DEC_HIDDEN_DIM=32
SEED=42
OPTIMIZER='adam'
ACCUM_STEPS=1
LR=1e-4
MOMENTUM=0.9
BATCH_SIZE=128
VAL_BATCH_SIZE=128
SAVE_BEST='True'

python3 src/main.py\
    --parameter_dir $PARMETER_DIR\
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