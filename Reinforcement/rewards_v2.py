from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import spacy
import nltk
import csv
import os
import pickle
import sys
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

sys.path.append('comet-commonsense')

import numpy as np
from operator import itemgetter
import json
import time
from collections import OrderedDict
from comet_generate import get_comet_prediction

from pathlib import Path
import tensorflow as tf
import torch
import texar as tx
from fast_bert.prediction import BertClassificationPredictor
from data_utils import utils
#from bert_embedding import BertEmbedding
from sklearn.metrics.pairwise import cosine_similarity



#bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased') #BertEmbedding()


SCORE_PATH = 'comet-commonsense/precomputed_similarities/'

with open(os.path.join(SCORE_PATH, 'word2index.pkl'), 'rb') as f:
    word2ind = pickle.load(f)
with open(os.path.join(SCORE_PATH, 'index2word.pkl'), 'rb') as f:
    ind2word = pickle.load(f)
with open(os.path.join(SCORE_PATH, 'glove/glove_embeddings.pkl'), 'rb') as f:
    glove_embeddings = pickle.load(f)

EMOTION_MAP = {'sadness':0, 'neutral':1, 'joy':2, 'fear': 3, 'anger': 4}


def tokenize_lemma(text, lemma=True):
    res = []

    for sent in text:
        sent = sent.strip('.')
        doc = nlp(sent)
        if lemma:
            filtered_sent = [i.lemma_ for i in doc if not nlp.vocab[i.text].is_stop]
        else:
            filtered_sent = [i.text for i in doc if not nlp.vocab[i.text].is_stop]
        res.append(filtered_sent)

    return res

def compute_emotion_accuracy(y_pred, y_true):
    arc_emotion_acc = (y_pred == y_true).all(axis=(1,2)).mean()
    segment_emotion_acc = (y_pred == y_true).all(axis=(2)).mean()

    return arc_emotion_acc, segment_emotion_acc

def compute_per_arc_accuracy(y_pred, y_true, dir_path):
    # load arc indices for the 10 most common arc (same in train and test)
    indices_dict = json.load(open(os.path.join(dir_path, 'arc_indices.json'),'r'))

    arc_acc = {}
    for k,v in indices_dict.items():
        y_pred_filtered = y_pred[v, :, :]
        y_true_filtered = y_true[v, :, :]
        arc_emo, _ = compute_emotion_accuracy(y_pred_filtered, y_true_filtered)
        arc_acc[k] = arc_emo

    print(arc_acc)
    return arc_acc


def sent_vectorizer(sent, model, dim=100):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass

    if sent_vec == []:
        return np.random.normal(scale=0.6, size=(dim,))

    return np.asarray(sent_vec) / numw

def compute_edit_distance(y_pred, y_true, unique_ids=None, batch_normalize=False):
    # with open('computed_similarities.pickle', 'rb') as handle:
    #     sim_matrix = pickle.load(handle)
    sim_matrix = np.load(os.path.join(SCORE_PATH, 'glove/similarity_matrix.npy'))
    embedding_matrix = np.load(os.path.join(SCORE_PATH, 'glove/embedding_matrix.npy'))
    basic2comet = {'anger': 'angry', 'fear': 'scared', 'joy': 'happy', 'sadness': 'sad', 'neutral': 'neutral'}
    reward = []
    # max(sim_matrix[word2ind[y_hat[1]]][word2ind[y[1]]], sim_matrix[word2ind[y_hat[2]]][word2ind[y[1]]], sim_matrix[word2ind[y_hat[3]]][word2ind[y[1]]]) + \

    batch_size = len(y_pred)
    flag_save = False
    if unique_ids is None:
        unique_ids = list(range(batch_size))
    for j in range(batch_size):
        y_hat = y_pred[j] #.split()
        if len(y_hat) < 3:
            reward.append(0)
            continue
        y = y_true[unique_ids[j]] #.split()
        y = [basic2comet[i] for i in y]
        y_hat = y_hat[:5]
        for i, react in enumerate(y_hat):
            if react not in word2ind:
                flag_save = True
                word2ind[react] = len(ind2word)
                ind2word[word2ind[react]] = react
                # emb = np.mean(np.array(bert_embedding([react])[0][1]), axis=0).reshape(1,-1)
                emb = sent_vectorizer(react.split(), glove_embeddings).reshape(1,-1)
                sim_matrix = np.append(sim_matrix, cosine_similarity(emb, embedding_matrix), axis=0)
        sub_reward = sim_matrix[word2ind[y_hat[0]]][word2ind[y[0]]] + \
            max([sim_matrix[word2ind[mid_arc]][word2ind[y[1]]] for mid_arc in y_hat[1:-1] ]) + \
                     sim_matrix[word2ind[y_hat[-1]]][word2ind[y[2]]]
        reward.append(sub_reward/3)


    if flag_save:
        np.save(os.path.join(SCORE_PATH, 'glove/similarity_matrix.npy'), sim_matrix)
        with open(os.path.join(SCORE_PATH, 'word2index.pkl'), 'wb') as f:
            pickle.dump(word2ind, f)
        with open(os.path.join(SCORE_PATH, 'index2word.pkl'), 'wb') as f:
            pickle.dump(ind2word, f)

    reward = np.array(reward)
    if batch_normalize:
        reward = np.mean(reward)
    return reward # shape (bs,)

def flatten_clf_res(x):
    " flatten the classifier result in one single list"
    prediction = [[i[1] for i in j] for j in x]
    prediction = [item for sublist in prediction for item in sublist]
    feat = [prediction[i:i + 15] for i in range(0, len(prediction), 15)] # len = bs

    return np.array(feat) # [bs, 15]

def get_emotion_dist(predictor, all_story, preprint=False):
    batch_size = len(all_story)  # all_story is list (bs) of list (3/5)
    # arc_len = len(all_story[0])

    flatten_stories = [i for sublist in all_story for i in sublist]
    prediction = flatten_clf_res(predictor.predict_batch(flatten_stories))
    assert (prediction.shape[0] == batch_size)

    return prediction

def compute_emotion_distance(y_pred, y_true, batch_normalize=False, reshape=True, method=None):
    # arc_len = y_pred.shape[1]
    batch_size = y_pred.shape[0]
    assert (y_pred.shape == y_true.shape)

    if method == 'clf_max':  # get the distance only on argmax
        y_pred = np.reshape(y_pred, (batch_size, 3, -1))
        y_true = np.reshape(y_true, (batch_size, 3, -1))

        y_pred_mx = np.zeros_like(y_pred)
        y_true_mx = np.zeros_like(y_true)

        y_pred_mx[np.arange(len(y_pred)), y_pred.argmax(1)] = y_pred[np.arange(len(y_pred)), y_pred.argmax(1)]
        y_true_mx[np.arange(len(y_true)), y_true.argmax(1)] = y_true[np.arange(len(y_true)), y_true.argmax(1)]

        dist = np.linalg.norm(y_pred_mx - y_true_mx, axis=(1, 2))

    else:
        dist = np.linalg.norm(y_true - y_pred, axis=1)

    if batch_normalize:
        dist = np.mean(dist)

    if method == 'negative':
        return -dist

    similarity_score = 1 / (1 + dist)

    return similarity_score  # shape (bs,) or scalar if batch_normalize=True

def get_emotion_prob(y_pred, arc_true, unique_ids=None, batch_normalize=False):
    batch_size = y_pred.shape[0]
    # assert (y_pred.shape[0] == len(arc_true))

    y_pred = np.reshape(y_pred, (batch_size, 3, -1)) # [bs, 3, 5]

    if unique_ids is None:
        unique_ids = list(range(batch_size))
    arc_true = list(itemgetter(*unique_ids)(arc_true))
    arc_true_ind = [[EMOTION_MAP[i] for i in sub] for sub in arc_true] # get indices of ground-truth emotion
    arc_true_ind = np.expand_dims(np.array(arc_true_ind), axis=2) # unsqueeze third dim
    y_prob = np.take_along_axis(y_pred, arc_true_ind, axis=2)

    prob_reward = np.sum(y_prob,axis=1)/3
    prob_reward = prob_reward.squeeze(1)
    if batch_normalize:
        prob_reward = np.mean(prob_reward)

    return prob_reward

def _get_text(proc, gen_result, gen_len=None):
    # gts_label shape [bs, 3*4]

    _samples = []
    if gen_len is not None:
        for i, l in zip(gen_result, gen_len):
            #  delete padding
            _samples.append(i[:l].tolist()) # list of list len = batch_size
    else:
        _samples.extend(h.tolist() for h in gen_result)
    _samples = utils.list_strip_eos(_samples, eos_token=proc.encoder['<|endoftext|>'])

    _all_text = []
    for s in _samples:
        s_text = proc.decode(s)
        s_text = s_text.replace('\n', ' ').strip("| ")
        _all_text.append(s_text)
    _all_text = tx.utils.strip_eos(_all_text, eos_token='<|endoftext|>')

    return _samples, _all_text

def format_generated_stories_for_clf(text_list, method=None):
    clf_input = []
    for txt in text_list:
        sample_story = nltk.sent_tokenize(txt) #should be list of len 5
        if method == 'clf':
            if len(sample_story) == 0:
                sample_story = ["", "", ""]
            elif len(sample_story) > 5:
                sample_story = [sample_story[0], ' '.join(sample_story[j] for j in range(1, 4)), sample_story[4]]
            elif len(sample_story) > 1:
                sample_story = [sample_story[0], ' '.join(j for j in sample_story[1:-1]), sample_story[-1]]
            else:
                sample_story = [sample_story[0], sample_story[0], sample_story[0]]

        clf_input.append(sample_story[:5])

    return clf_input # list of list len = bs , len = arc_size/5

def generate_all_valid_sample_dict(predictor, ids, stories, method=None):
    result_dict = {}
    # generate comet reacts
    if method == 'comet':
        reacts = get_comet_prediction(stories)
        # print(ids, reacts)
        for ind, item in zip(ids, reacts):
            result_dict[ind] = item
    # obtain emotion dist
    else:
        emotion_dist = get_emotion_dist(predictor, stories)
        for ind, item in zip(ids, emotion_dist):
            result_dict[ind] = item
    return result_dict


def get_reward(predictor, gen_result, ids, gts_arc, batch_normalize=False, method=None):
    # print(gts_label.shape)
    if method == 'comet':
        comet_prediction = get_comet_prediction(gen_result)
        score = compute_edit_distance(comet_prediction, gts_arc, ids)
    elif method == 'clf_prob':
        emotion_dist = get_emotion_dist(predictor, gen_result)
        score = get_emotion_prob(emotion_dist, gts_arc, ids)

    return score

def emotion_evaluation(path, arc_path=None, binarized=True, method=None):
    """
    for test after finishing training
    """

    #load emotion classifier
    LABEL_PATH = "emotion_classifier/"
    MODEL_PATH = "emotion_classifier/checkpoint/bert/model_out/"

    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=LABEL_PATH,  # location for labels.csv file
        multi_label=True,
        model_type='bert',
        do_lower_case=True)


    # load and process generated file[]
    if os.path.exists("numpy_files_v3/generated_em_dist_rl_fine.npy"):
        print("Loading computed emotion dist for generated stories...")
        generated_emotion_scores = np.load("numpy_files_v3/generated_em_dist_rl_fine.npy") 
    else:
        print("Start loading and processing generated stories...")
        _all_text = []
        with open(path) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                # trim prefix context and suffix EOS
                txt = row[1].strip(" | ")
                ind = txt.find(" <|endoftext|>")
                txt = txt[:ind] if ind != -1 else txt
                _all_text.append(txt)

        clf_input = []
        comet_input = []
        for txt in _all_text:
            sample_story = nltk.sent_tokenize(txt) #should be list of len 5
            comet_input.append(sample_story)
            # for some reason the model rarely generates not exactly 5 sentences
            if len(sample_story) == 0:
                sample_story = ["", "", ""]
            elif len(sample_story) > 5:
                sample_story = [sample_story[0], ' '.join(sample_story[j] for j in range(1, 4)), sample_story[4]]
            elif len(sample_story) > 1:
                sample_story = [sample_story[0], ' '.join(j for j in sample_story[1:-1]), sample_story[-1]]
            else:
                sample_story = [sample_story[0], sample_story[0], sample_story[0]]

            clf_input.append(sample_story[:5])

        print("Start classifying generated stories...")
        generated_emotion_scores = get_emotion_dist(predictor, clf_input, preprint=False) # np array (data_size, 3 * 5)
        np.save("numpy_files_v3/generated_em_dist_rl_base_k40.npy", generated_emotion_scores)
        print("Classification finished !")



    if arc_path is not None:
        test_arc = [i.strip().split() for i in open(arc_path)]
        print("Start computing emotion probability score")
        emo_prob_score = get_emotion_prob(generated_emotion_scores, test_arc, batch_normalize=True)
        print("clf_prob score: ", emo_prob_score)
        metrics.update({"classifier probablity score: ": emo_prob_score})

    if binarized:
        data_size = len(test_arc)
        generated_emotion_scores = np.reshape(generated_emotion_scores, (data_size, 3, -1))

        generated_emotion_scores_bn = (generated_emotion_scores.max(axis=-1, keepdims=1) == generated_emotion_scores).astype(float)

        if os.path.exists(arc_path[:-4]+".npy"):
            true_emotion_scores_bn = np.load(arc_path[:-4]+".npy")
        else:
            true_emotion_scores_bn = np.zeros_like(generated_emotion_scores)
            assert(generated_emotion_scores.shape[:2] == (len(test_arc), len(test_arc[0])))
            for i in range(true_emotion_scores_bn.shape[0]):
                for j in range(true_emotion_scores_bn.shape[1]):
                    true_emotion_scores_bn[i][j][EMOTION_MAP[test_arc[i][j]]] = 1.0

            np.save(arc_path[:-4]+".npy", true_emotion_scores_bn)


        arc_emotion_accuracy, seg_emotion_accuracy = compute_emotion_accuracy(generated_emotion_scores_bn, true_emotion_scores_bn)
        print("arc_emotion_accuracy: {}\n segment_emotion_accuracy: {}".format(arc_emotion_accuracy, seg_emotion_accuracy))
        metrics.update({"arc_acc": arc_emotion_accuracy, "segment_acc": seg_emotion_accuracy})

        dic_dir = os.path.dirname(label_path)
        per_arc_accuracy = compute_per_arc_accuracy(generated_emotion_scores_bn, true_emotion_scores_bn, dic_dir)
        metrics.update(per_arc_accuracy)

    # compute comet-based emotion evaluation metric (Ec-Em) 
    if arc_path is not None:
        test_arc_file = [i.strip().split() for i in open(arc_path)]
        print("Start generating comet inferences ...")
        comet_prediction = get_comet_prediction(comet_input)
        print("Finished generating comet inferences ...")
        comet_score = compute_edit_distance(comet_prediction, test_arc_file, batch_normalize=True)
        print("comet score: {}".format(comet_score))
        metrics.update({"comet_score: ": comet_score})


    return metrics

