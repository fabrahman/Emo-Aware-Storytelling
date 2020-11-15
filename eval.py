import texar as tx
import nltk

import numpy as np
from collections import Counter
import sys, os, pickle
sys.path.append('../..')

from LIB.EVAL.meteor import Meteor
from LIB.EVAL.bleu import compute_bleu, diverse_bleu
from LIB.EVAL.rouge import compute_rouge_L
from Reinforcement.rewards_v2 import compute_emotion_distance, get_emotion_prob, sent_vectorizer

#from bert_embedding import BertEmbedding
from sklearn.metrics.pairwise import cosine_similarity


#bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')

SCORE_PATH = 'comet-commonsense/precomputed_similarities/'

with open(os.path.join(SCORE_PATH, 'word2index.pkl'), 'rb') as f:
    word2ind = pickle.load(f)
with open(os.path.join(SCORE_PATH, 'index2word.pkl'), 'rb') as f:
    ind2word = pickle.load(f)
with open(os.path.join(SCORE_PATH, 'glove/glove_embeddings.pkl'), 'rb') as f:
    glove_embeddings = pickle.load(f)

def compute_edit_distance_v2(y_pred, y_true, batch_normalize=True):
    # with open('computed_similarities.pickle', 'rb') as handle:
    #     sim_matrix = pickle.load(handle)
    sim_matrix = np.load(os.path.join(SCORE_PATH, 'glove/similarity_matrix.npy'))
    embedding_matrix = np.load(os.path.join(SCORE_PATH, 'glove/embedding_matrix.npy'))
    basic2comet = {'anger': 'angry', 'fear': 'scared', 'joy': 'happy', 'sadness': 'sad', 'neutral': 'neutral'}
    reward = []

    # batch_size = len(y_pred)
    flag_save = False
    for k, v in y_pred.items():
        y_hat = v #y_pred[j] #.split()
        if len(y_hat) < 3:
            reward.append(0)
            continue
        y = y_true[k] #.split()
        y = [basic2comet[i] for i in y]
        for i, react in enumerate(y_hat[:5]):
            if react not in word2ind:
                flag_save = True
                word2ind[react] = len(ind2word)
                ind2word[word2ind[react]] = react
                # emb = np.mean(np.array(bert_embedding([react])[0][1]), axis=0).reshape(1,-1)
                emb = sent_vectorizer(react.split(), glove_embeddings).reshape(1, -1)
                ### Bert version
                # if not np.isnan(emb.any()) and np.isfinite(emb.all()):
                #     sim_matrix = np.append(sim_matrix, cosine_similarity(emb, embedding_matrix), axis=0)
                # else:
                #     sim_matrix = np.append(sim_matrix, np.zeros((1, sim_matrix.shape[1])), axis=0)
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

def evaluate_full(gts, gen_res, gen_dict, gts_arc, emo_feat, method=None):
    assert (len(gts) == len(gen_res))
    reward = 0
    if method == "comet":
        assert (len(gen_dict) == len(gts_arc))
        reward = compute_edit_distance_v2(gen_dict, gts_arc)
    elif method == "clf":
        pred_dist = np.zeros_like(emo_feat)
        for k, v in gen_dict.items():
            pred_dist[k,:] = v
        reward = compute_emotion_distance(pred_dist, emo_feat, batch_normalize=True, method=method)
    elif method == 'clf_prob':
        pred_dist = np.zeros_like(emo_feat)
        for k, v in gen_dict.items():
            pred_dist[k,:] = v
        reward = get_emotion_prob(pred_dist, gts_arc, batch_normalize=True)


    translation_corpus = gen_res
    reference_corpus = gts
    rouges = []
    meteor = Meteor()
    bleu = compute_bleu([[j.split()] for j in reference_corpus], [i.split() for i in translation_corpus])
    mete = meteor.compute_score([[r] for r in reference_corpus], translation_corpus)


    return {"bleu": bleu[0] * 100, "meteor": mete[0] * 100, "best_reward": reward}
