# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utils of data preprocessing for GPT2 training.
"""

import os
import collections
import pickle
from operator import itemgetter
from pathlib import Path
import csv
import nltk
import tensorflow as tf
import torch
from fast_bert.prediction import BertClassificationPredictor





# pylint: disable=invalid-name, too-many-arguments

class InputExample(object):

    def __init__(self, x1, x4, comet, yy=None):
        self.x1 = x1
        self.x4 = x4
        self.comet = comet


def flatten_clf_res(x):
    " flatten the classifier result in one single list"
    res = []
    label = []
    for phase_score in x:
        res += [i[1] for i in phase_score]
        label.append(max(phase_score, key=itemgetter(1))[0])  # get the prominent emotion for each phase

    return res, label # res len 12 | label len 3

def _truncate_seqs(x1, x4, max_length, encoder):
    while True:
        ids = encoder.encode(x1 + ' | ' + x4 + ' ')
        if len(ids) <= max_length:
            break
        # print("***", len(ids))
        x4_ = x4.split()
        x4 = ' '.join(x4_[:-1])
    return x4

def _truncate_seqs_v2(x1, x3, max_length, encoder):
    while True:
        ids = encoder.encode(x1 + ' | ' + x3 + ' ')
        if len(ids) <= max_length:
            break
        # print("***", len(ids))
        x3_ = x3.split()
        x3 = ' '.join(x3_[:-1])
    return x3

def process_emotion_single(predictor, story):

    x_sentences = nltk.sent_tokenize(story.strip(" <|endoftext|>"))
    mid_sentences = x_sentences[1:-1]
    x_sentences = [x_sentences[0], ' '.join(j for j in mid_sentences), x_sentences[-1]] # list of length 3
    prediction = predictor.predict_batch(x_sentences)
    prediction, labels = flatten_clf_res(prediction)

    middle_pred = predictor.predict_batch(mid_sentences)
    middle_labels = flatten_clf_res(middle_pred)[1]

    return prediction, labels, middle_labels

def process_single_example(example, max_seq_length, encoder):
    x1 = example.x1
    x4 = example.x4
    comet = example.comet

    ### Add emotion labels to input title
    x1 = comet + ' <$> ' + x1


    x4 = _truncate_seqs(x1, x4, max_seq_length - 2, encoder)

    mask_text = 'Unknown .'
    special = encoder.encoder['<|endoftext|>']

    x1_ids = encoder.encode(x1)

    ### Title2Story  ----> Finetune
    x1x4 = x1 + ' | ' + x4
    x1x4_ids = encoder.encode(x1x4)
   # print(x1x4)

    len_x1 = len(x1_ids)
    len_x1x4 = len(x1x4_ids)


    while len(x1_ids) < max_seq_length:
        x1_ids.append(special)
    while len(x1x4_ids) < max_seq_length:
        x1x4_ids.append(special)


    feature = {
        "x1_ids": x1_ids,
        "x1_len": len_x1,
        "x1x4_ids": x1x4_ids,
        "x1x4_len": len_x1x4,
    }

    return feature


def read_raw_data_v2(path, mode):
    def _read_file(fn):
        with open(fn, 'r') as fin:
            lines = [line.strip() for line in fin]
        return lines

    def _get_fn(field):
        return os.path.join(path, '%s_%s.txt' % (mode, field))

    all_x1 = _read_file(_get_fn('x1'))
    all_x4 = _read_file(_get_fn('x4'))
    all_comet = _read_file(_get_fn('mapped'))

    print('#examples: %d' % len(all_x1))

    return [
        InputExample(
            x1=x1,
            x4=x4,
            comet=comet # comet annotations
        )
        for x1, x4, comet in zip(all_x1, all_x4, all_comet)
    ]


def file_based_convert_examples_to_features_v2(
        examples, max_seq_length, encoder, output_file, mode, verbose=False):
    """Converts a set of examples to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

#    with open(os.path.join(os.path.dirname(output_file), "x4_emo_features.{}".format(mode)), 'rb') as fp:
#        emotion_feats = pickle.load(fp)

    for (i, example) in enumerate(examples):
        if i % 5000 == 0:
            print("{} of {} is processed!".format(i, len(examples)))

#        emotion_fea = emotion_feats[i]


        fea = process_single_example(
            example, max_seq_length, encoder)

        # if verbose:
        #     print(fea["x1x2yx1xx2_len"])

        def _create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        def _create_floats_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

        def _create_string_feature(values):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.encode('utf-8')])) # m.encode('utf-8') for m in values

        features = collections.OrderedDict()
        features["x1_ids"] = _create_int_feature(fea["x1_ids"])
        features["x1_len"] = _create_int_feature([fea["x1_len"]])
        features["x1x4_ids"] = _create_int_feature(fea["x1x4_ids"])
        features["x1x4_len"] = _create_int_feature([fea["x1x4_len"]])
#        features["x4_emo"] = _create_floats_feature(emotion_fea)
        features["arc_label"] = _create_string_feature(example.comet)
        features["unique_id"] = _create_int_feature([i])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())



def prepare_TFRecord_data_v2(data_dir, max_seq_length, encoder, output_dir):
    """
    Args:
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the TFRecord files in.
    """

    train_examples = read_raw_data_v2(data_dir, mode='train')
    print('##train examples: %d' % len(train_examples))
    train_file = os.path.join(output_dir, "train.tf_record")
    file_based_convert_examples_to_features_v2(
        train_examples, max_seq_length, encoder, train_file, mode='train')

    eval_examples = read_raw_data_v2(data_dir, mode='dev')
    print('##dev examples: %d' % len(eval_examples))
    eval_file = os.path.join(output_dir, "dev.tf_record")
    file_based_convert_examples_to_features_v2(
       eval_examples, max_seq_length, encoder, eval_file, mode='dev')

    test_examples = read_raw_data_v2(data_dir, mode='test')
    print('##test examples: %d' % len(test_examples))
    test_file = os.path.join(output_dir, "test.tf_record")
    file_based_convert_examples_to_features_v2(
       test_examples, max_seq_length, encoder, test_file, mode='test', verbose=True)


