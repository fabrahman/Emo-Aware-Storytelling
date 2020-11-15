#!/bin/bash

# middle size
# fits to x1x4

gpu_id=1
is_train=1
topk=40

test_checkpoint="output/clf_prob_rl/model_best.ckpt"

loaded_gpt="output/orig/gpt2_model.ckpt"

output_dir="output/clf_prob_rl"
config_train=config_train_128_comet

config_model=configs.config_model_345M
pretrained_model_dir=gpt2_pretrained_models/model_345M
pretrain_checkpoint=gpt2_pretrained_models/model_345M/model.ckpt

rl_method="clf_prob" # or "comet"

mkdir -p ${output_dir}
cp $0 ${output_dir}
cp configs/${config_train}.py ${output_dir}


if [ "$is_train" = 1 ]; then ## train

  CUDA_VISIBLE_DEVICES=${gpu_id}  \
  python train_rl.py \
    --config_model=${config_model} \
    --checkpoint=${test_checkpoint} \
    --pretrained_model_dir=${pretrained_model_dir} \
    --config_train=configs.${config_train} \
    --pretrain_checkpoint=${pretrain_checkpoint} \
    --output_dir=${output_dir} \
    --do_train \
    --sc_rl \
    --rl_method=${rl_method}
# #    --checkpoint=${test_checkpoint} \

else ## test

  CUDA_VISIBLE_DEVICES=${gpu_id}  \
  python generate_all.py  \
    --config_model=${config_model} \
    --pretrained_model_dir=${pretrained_model_dir} \
    --config_train=configs.${config_train} \
    --checkpoint=${test_checkpoint} \
    --output_dir=${output_dir} \
    --do_test \
    --finetune \
    --top_k=${topk}
#    --bpe_loss

fi
