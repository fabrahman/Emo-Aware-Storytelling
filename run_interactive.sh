#!/bin/bash

gpu_id=0 # 1,0
is_train=0
topk=40

test_checkpoint="output/clf_prob_rl/model_best.ckpt"


config_model=configs.config_model_345M
pretrained_model_dir=/home/hannah/Counterfactual-StoryRW/gpt2_pretrained_models/model_345M
pretrain_checkpoint=/home/hannah/Counterfactual-StoryRW/gpt2_pretrained_models/model_345M/model.ckpt

rl_method="clf_prob"

#mkdir -p ${output_dir}
#cp $0 ${output_dir}
#cp configs/${config_train}.py ${output_dir}

CUDA_VISIBLE_DEVICES=${gpu_id}  \
python interactive_generation.py  \
  --config_model=${config_model} \
  --pretrained_model_dir=${pretrained_model_dir} \
  --checkpoint=${test_checkpoint} \
  --is_interactive


