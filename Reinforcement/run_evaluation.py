import argparse
import json
from typing import List

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from rewards_v2 import emotion_evaluation
import spacy
import tqdm
import numpy as np
# import rouge
# import edlib
import os
import pandas as pd
import re
import glob
import sys



def main(pred_file, arc_file):

    metrics = {}
    print("Evaluate Emotion Similarity ... ")
    metrics.update(emotion_evaluation(pred_file, arc_file))


    return metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_evaluation.py',
        usage='%(prog)s gold_annotations predictions',
        description='Evaluate story'
    )

    parser.add_argument('--pred-file', type=str,
                        dest="pred_file",
                        help='Location of prediction file. Usually named test_samples_*.tsv',
                        default=None)

    parser.add_argument('--all-preds-dir', type=str,
                        dest="all_preds_dir",
                        help='Location of prediction file. Usually named *_pred.txt',
                        default=None)

    parser.add_argument('--arc-file', type=str,
                        dest="arc_file",
                        help='Location of extracted emotion label for [dev/test/train].txt. Usually named as [train/test/dev]_mapped.txt',
                        default=None)

    parser.add_argument('--output_file', type=str,
                        dest="output_file",
                        help='')

    args = parser.parse_args()

    # Run seed selection if args valid
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")

    assert args.all_preds_dir is not None or args.pred_file is not None

    all_metrics = {}
    if args.all_preds_dir is not None:
        for f in glob.iglob(args.all_preds_dir + "/*/*.txt"):
            print("Processing file {}".format(f))
            metrics = main(f, args.gold_file, args.bert_model)
            model_name = os.path.basename(f).split(".")[0]
            all_metrics[model_name] = metrics

    else:
        all_metrics = main(args.pred_file, args.arc_file)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(all_metrics))
        f.close()
