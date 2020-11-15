from transformers import BertTokenizer
from pathlib import Path
import torch

from box import Box
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import apex
from sklearn.model_selection import train_test_split

import datetime

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc, accuracy_jaccord


from sklearn.metrics import f1_score
from torch import Tensor

def F1_macro(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True, thresh:float = 0.4, average = 'macro', sample_weight = None):
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    return f1_score(y_true, y_pred, average = average, sample_weight = sample_weight)

def F1_micro(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True):
    return F1_macro(y_pred, y_true, sigmoid = sigmoid, average = 'micro')




LOG_PATH=Path('Affect-in-Tweets/ROCStories_train_data/old_model/checkpoint/bert/logs/')
DATA_PATH = LABEL_PATH = "Affect-in-Tweets/ROCStories_train_data/old_model/" 

# model_state_dict = None

# BERT_PRETRAINED_PATH = Path('../../bert_models/pretrained-weights/cased_L-12_H-768_A-12/')
# BERT_PRETRAINED_PATH = Path('../../bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')
# BERT_PRETRAINED_PATH = Path('../../bert_fastai/pretrained-weights/uncased_L-24_H-1024_A-16/')
FINETUNED_PATH = Path('Affect-in-Tweets/5_emotion_data/checkpoint/bert/model_out/pytorch_model.bin')
# FINETUNED_PATH = None
#model_state_dict = torch.load(FINETUNED_PATH)

LOG_PATH.mkdir(exist_ok=True)

OUTPUT_PATH = Path("Affect-in-Tweets/ROCStories_train_data/old_model/checkpoint/bert/") 
OUTPUT_PATH.mkdir(exist_ok=True)

args = Box({
    "run_text": "multilabel emotion classification",
    "train_size": -1,
    "val_size": -1,
    "log_path": LOG_PATH,
    "full_data_dir": DATA_PATH,
    "data_dir": DATA_PATH,
    "task_name": "finetune_on_comet_output",
    "no_cuda": False,
#     "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": OUTPUT_PATH,
    "max_seq_length": 128,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 6,
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "num_train_epochs": 5,
    "warmup_proportion": 0.0,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "fp16_opt_level": "O1",
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "warmup_steps": 500,
    "logging_steps": 50,
    "eval_all_checkpoints": True,
    "overwrite_output_dir": True,
    "overwrite_cache": False,
    "seed": 42,
    "loss_scale": 128,
    "task_name": 'emotion',
    "model_name": 'bert-large-cased', #'xlnet-base-cased',
    "model_type": 'bert' #'xlnet'
})





import logging
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()

logger.info(args)


device = torch.device(1)
# if torch.cuda.device_count() > 1:
#     args.multi_gpu = True
# else:
args.multi_gpu = False
    
label_cols = ['anger', 'fear','joy', 'sadness', 'neutral']

databunch_bert_fine_comet = BertDataBunch(args['data_dir'], LABEL_PATH, args.model_name, train_file='train.csv', val_file='val.csv',
                          text_col="Tweet", label_col=label_cols,
                          batch_size_per_gpu=args['train_batch_size'], max_seq_length=args['max_seq_length'], 
                          multi_gpu=args.multi_gpu, multi_label=True, model_type=args.model_type)





### trained with comet main character annotations on bert large finetunes on tweets / 5 epochs

metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'accuracy_jaccord', 'function': accuracy_jaccord})
metrics.append({'name': 'f1_macro', 'function': F1_macro})
metrics.append({'name': 'f1_micro', 'function': F1_micro})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'fbeta', 'function': fbeta})

bert_learner_5_finetuned = BertLearner.from_pretrained_model(databunch_bert_fine_comet, args.model_name, metrics=metrics, 
                                            device=device, logger=logger, output_dir=args.output_dir, 
                                            finetuned_wgts_path=FINETUNED_PATH, warmup_steps=args.warmup_steps,
                                            multi_gpu=False, is_fp16=args.fp16, 
                                            multi_label=True, logging_steps=0)

bert_learner_5_finetuned.fit(args.num_train_epochs, args.learning_rate, validate=True)
bert_learner_5_finetuned.save_model()
