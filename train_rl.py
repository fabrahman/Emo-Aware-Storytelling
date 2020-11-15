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
"""Example of fine-tuning OpenAI GPT-2 language model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import importlib
import numpy as np
import tensorflow as tf
import texar as tx
import copy
import pickle

sys.path.append('../comet-commonsense')

from data_utils import model_utils, processor, utils

# import torch
# from pathlib import Path
from fast_bert.prediction import BertClassificationPredictor
from Reinforcement.rewards_v2 import get_reward, format_generated_stories_for_clf, _get_text, generate_all_valid_sample_dict
from pathlib import Path

from eval import evaluate_full


# pylint: disable=invalid-name, too-many-locals, too-many-statements, no-member
# pylint: disable=invalid-name, too-many-locals, too-many-statements, no-member
# pylint: disable=too-many-branches

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None,
                    "Model checkpoint to resume training or for test.")
flags.DEFINE_string("pretrain_checkpoint",
                    "gpt2_pretrained_models/model_117M/model.ckpt",
                    "OpenAI pretrained model checkpoint. Ignored if "
                    "'--checkpoint' is specified.")
flags.DEFINE_string("pretrained_model_dir", "gpt2_pretrained_models/model_117M",
                    "The directory of pretrained model, for loading vocabuary, "
                    "etc.")
flags.DEFINE_float("temperature", 0.7,
                   "Softmax temperature for top-k sample decoding. Must be "
                   "strictly greater than 0. Defaults to 0.7.")
flags.DEFINE_integer("top_k", 40,
                     "The number of top most likely candidates from a vocab "
                     "distribution.")
flags.DEFINE_string("config_train", "configs.config_train",
                    "Configurations of GPT-2 training, including data and "
                    "optimization hyperparameters.")
flags.DEFINE_string("config_type", "texar",
                    "The configuration file type. Set to 'json' if the GPT-2 "
                    "config file is in the same type of the official GPT-2 "
                    "config file. Set to 'texar' if GPT-2 config file is in "
                    "Texar type.")
flags.DEFINE_string("config_model", "configs.config_model",
                    "The model configuration file to configure the model. "
                    "The config file type is define by the 'config_type',"
                    "it be of texar type or json type."
                    "For '--config_type=json', set the json config file path"
                    "like: '--config_model gpt2_pretrained_models/model_117M/"
                    "hparams.json';"
                    "For '--config_type=texar', set the texar config file "
                    "like: '--config_model configs.config_model'.")
flags.DEFINE_string("output_dir", "output/remove_space/",
                    "The output directory where the model checkpoints will be "
                    "written.")
flags.DEFINE_string("clf_label_dir", "emotion_classifier/", #ROC_cloze_data
                    " The directory where emotion labels for emotion classifier are"
                    "written in a file named labels.csv.")
flags.DEFINE_string("clf_output_dir", "emotion_classifier/checkpoint/bert/model_out/", ##ROC_cloze_data
                    "The output directory where the emotion classifier checkpoints are "
                    "saved.")
flags.DEFINE_string("rl_method", "clf",
                    "train rl-clf or comet")
flags.DEFINE_string("best_model", "emotion",
                    "save best model based on which metric during validation?")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_test", False, "Whether to run test on the test set.")
flags.DEFINE_bool("distributed", False, "Whether to run in distributed mode.")
flags.DEFINE_bool("finetune", False, "Whether to test on finetune mode.")
flags.DEFINE_bool("sc_rl", False, "Whether to train with self-critical RL")
flags.DEFINE_bool("beam", False, "Whether to do a beam search for inference?")

config_train = importlib.import_module(FLAGS.config_train)


def _log(msg, log_fn=None):
    tf.logging.info(msg)
    if log_fn is None:
        log_fn = os.path.join(FLAGS.output_dir, config_train.name, 'log.txt')
    with open(log_fn, 'a') as flog:
        flog.write(msg + '\n')

def _ids_to_text(ids, proc):
    eos_token_id = proc.encoder['<|endoftext|>']

    if ids[0] == eos_token_id:
        ids = ids[1:]
    text = proc.decode(ids)
    return text

def _fix(input_ids, eos_token_id):
    input_ids = copy.deepcopy(input_ids)
    # for i in range(len(input_ids)):
    #     if bos_token_id is not None:
    #         input_ids[i] = [bos_token_id] + input_ids[i]
    #     if eos_token_id is not None:
    #         input_ids[i] = input_ids[i] + [eos_token_id]

    length = [len(ids) for ids in input_ids]
    max_length = max(length)
    for i in range(len(input_ids)):
        while len(input_ids[i]) < max_length:
            input_ids[i].append(eos_token_id)

    return np.array(input_ids), np.array(length)


# load trained emotion classifier
predictor = BertClassificationPredictor(
    model_path=FLAGS.clf_output_dir,
    label_path=FLAGS.clf_label_dir,  # location for labels.csv file
    multi_label=True,
    model_type='bert',
    do_lower_case=True)

def main(_):
    """
    Builds the model and runs
    """
    if FLAGS.distributed:
        import horovod.tensorflow as hvd
        hvd.init()

    tf.logging.set_verbosity(tf.logging.INFO)

    if len(config_train.name) > 0:
        output_dir = os.path.join(FLAGS.output_dir, config_train.name)
    else:
        output_dir = FLAGS.output_dir
    tx.utils.maybe_create_dir(output_dir)


    ## Loads GPT-2 model configuration

    if FLAGS.config_type == "json":
        gpt2_config = model_utils.transform_gpt2_to_texar_config(
            FLAGS.config_model)
    elif FLAGS.config_type == 'texar':
        gpt2_config = importlib.import_module(
            FLAGS.config_model)
    else:
        raise ValueError('Unknown config_type.')

    # Creates a data pre-processor for, e.g., BPE encoding
    proc = processor.get_encoder(FLAGS.pretrained_model_dir)
    end_token = proc.encoder['<|endoftext|>']

    max_decoding_length = config_train.max_decoding_length
    assert max_decoding_length <= gpt2_config.position_size, (
        "max_decoding_length should not be greater than position_size. "
        "{}>{}".format(max_decoding_length, gpt2_config.position_size))

    ## Loads data

    # Configures training data shard in distribued mode
    if FLAGS.distributed:
        config_train.train_hparam["dataset"]["num_shards"] = hvd.size()
        config_train.train_hparam["dataset"]["shard_id"] = hvd.rank()
        config_train.train_hparam["batch_size"] //= hvd.size()

    datasets = {}
    #if FLAGS.do_train:
    train_dataset = tx.data.TFRecordData(hparams=config_train.train_hparam)
    datasets['train'] = train_dataset
    #if FLAGS.do_eval:
    dev_dataset = tx.data.TFRecordData(hparams=config_train.dev_hparam)
    datasets['dev'] = dev_dataset
    #if FLAGS.do_test:
    test_dataset = tx.data.TFRecordData(hparams=config_train.test_hparam)
    datasets['test'] = test_dataset
    iterator = tx.data.FeedableDataIterator(datasets)
    batch = iterator.get_next()
    batch_size = tf.shape(batch['x1x4_ids'])[0]

    ## Builds the GPT-2 model
    vocab_size = gpt2_config.vocab_size

    word_embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size,
        hparams=gpt2_config.embed)

    pos_embedder = tx.modules.PositionEmbedder(
        position_size=gpt2_config.position_size,
        hparams=gpt2_config.pos_embed)

    # Ties output layer with input word embedding
    output_layer = tf.transpose(word_embedder.embedding, (1, 0))

    decoder = tx.modules.TransformerDecoder(
        vocab_size=vocab_size,
        output_layer=output_layer,
        hparams=gpt2_config.decoder)

    def _embedding_fn(ids, times):
        return word_embedder(ids) + pos_embedder(times)

    # For training
    def _get_recon_loss(ids, full_len, prefix_len=None, mask_prefix=True, do_print=False):
        ids = ids[:,:tf.reduce_max(full_len)]
        batch_size__ = tf.shape(ids)[0]
        seq_len = tf.fill([batch_size__], tf.shape(ids)[1])
        pos_embeds = pos_embedder(sequence_length=seq_len)
        input_embeds = word_embedder(ids) + pos_embeds

        # greedy output
        outputs = decoder(inputs=input_embeds, decoding_strategy='train_greedy')



        max_full_len = tf.reduce_max(full_len)
        ids = ids[:, :max_full_len]
        logits = outputs.logits[:, :max_full_len]

        if mask_prefix:
            loss_recon = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=ids[:, 1:],
                logits=logits[:, :-1, :],
                sequence_length=full_len-1,
                average_across_timesteps=False,
                sum_over_timesteps=False,
                average_across_batch=False,
                sum_over_batch=False)
            mask_recon = tf.sequence_mask(
                full_len-1,
                dtype=tf.float32)
            mask_recon_prefix = 1 - tf.sequence_mask(
                prefix_len-1,
                maxlen=max_full_len-1,#max_decoding_length-1,
                dtype=tf.float32)
            mask_recon = mask_recon * mask_recon_prefix

            if do_print:
                print_op_1 = tf.print(mask_recon)
                loss_recon_flat = tx.utils.reduce_with_weights(
                    tensor=loss_recon,
                    weights=mask_recon,
                    average_across_remaining=False,
                    sum_over_remaining=False,
                    average_across_batch=False)
                print_op_2 = tf.print(loss_recon_flat)
                with tf.control_dependencies([print_op_1, print_op_2]):
                    loss_recon = tx.utils.reduce_with_weights(
                        tensor=loss_recon,
                        weights=mask_recon,
                        average_across_remaining=True,
                        sum_over_remaining=False)
                return loss_recon, mask_recon, loss_recon_flat
            else:
                loss_recon = tx.utils.reduce_with_weights(
                    tensor=loss_recon,
                    weights=mask_recon,
                    average_across_remaining=True,
                    sum_over_remaining=False)
        else:
            loss_recon = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=ids[:, 1:],
                logits=logits[:, :-1, :],
                sequence_length=full_len-1,
                average_across_timesteps=True,
                sum_over_timesteps=False,
                average_across_batch=False,
                sum_over_batch=False)

        return loss_recon

    # For RL fine-tuning
    def _get_sample_story(context_ids, context_len):
        sample_output, sample_len = decoder(
            decoding_strategy='infer_sample',
            embedding = _embedding_fn,
            context=context_ids,
            context_sequence_length=context_len,
            max_decoding_length=max_decoding_length,
            end_token=end_token,
            softmax_temperature=FLAGS.temperature,
            mode=tf.estimator.ModeKeys.PREDICT)

        return sample_output, sample_len
        # return ids, batch_loss, ids_len

    def _get_sample_rolled(output, length, context_len):

        ids = output.sample_id
        ids = tx.utils.varlength_roll(ids, -context_len)  # final sample ids rolled
        ids_len = length - context_len
        ids = ids[:, :tf.reduce_max(ids_len)]

        return ids, ids_len

    def compute_batch_loss(output, sample_len, context_len):
        max_full_len = tf.reduce_max(sample_len)
        ids = output.sample_id[:, :max_full_len]
        logits = output.logits[:, :max_full_len] #(bs, sl, vocab)

        sampleLogprobs = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=ids[:,1:],
            logits=logits, 
            sequence_length=sample_len - 1, 
            average_across_timesteps=False,
            sum_over_timesteps=False,
            average_across_batch=False,
            sum_over_batch=False)

        mask = tf.sequence_mask(
            sample_len-1,
            dtype=tf.float32)
        mask_prefix = 1 - tf.sequence_mask(
            context_len-1,
            maxlen=max_full_len-1, #max_decoding_length-1,
            dtype=tf.float32)
        mask = mask * mask_prefix

        batch_loss = tx.utils.reduce_with_weights(
             tensor=sampleLogprobs,
             weights=mask,
             average_across_batch=False,
             average_across_remaining=True,
             sum_over_remaining=False)

        return batch_loss


    def _get_greedy_story(context_ids, context_len):

        greedy_res, greedy_len = decoder(
            decoding_strategy='infer_greedy',
            embedding=_embedding_fn,
            context=context_ids,
            context_sequence_length=context_len,
            max_decoding_length=max_decoding_length,
            end_token=end_token,
            mode=tf.estimator.ModeKeys.PREDICT)

        greedy_ids = tx.utils.varlength_roll(greedy_res.sample_id, -context_len)
        greedy_ids_len = greedy_len - context_len
        greedy_ids = greedy_ids[:, :tf.reduce_max(greedy_ids_len)]

        return greedy_ids, greedy_ids_len


    ## ROC Loss-1: ML loss
    x1_len = tf.placeholder(tf.int32, shape=[None], name='x1_len')
    x1x4_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1x4_ids')
    x1x4_len = tf.placeholder(tf.int32, shape=[None], name='x1x4_len')

    loss_fine = _get_recon_loss(x1x4_ids, x1x4_len, x1_len)

    x1_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1_ids')
    reward = tf.placeholder(tf.float32, shape=[None], name="reward")
    sampled_story = tf.placeholder(tf.int32, shape=[None, None], name="sampled_story") #smilar to sample_que
    sampled_story_len = tf.placeholder(tf.int32, shape=[None], name='sample_story_len')

    ## Loss-2: RL loss
    symbols_output, symbols_len = _get_sample_story(x1_ids, x1_len)
    symbols_rl, len_rl = _get_sample_rolled(symbols_output, symbols_len, x1_len)
    symbols_gr, len_gr = _get_greedy_story(x1_ids, x1_len)
    batch_loss_rl = _get_recon_loss(sampled_story, sampled_story_len, mask_prefix=False)
    rl_loss_fine = tf.reduce_mean(batch_loss_rl * reward)



    def _get_beam_ids(context_ids, context_len, target):
        # beam-search
        predictions = decoder(
            beam_width=5,
            length_penalty=config_train.length_penalty,
            embedding=_embedding_fn,
            context=context_ids,
            context_sequence_length=context_len,
            max_decoding_length=max_decoding_length,
            end_token=end_token,
            mode=tf.estimator.ModeKeys.PREDICT)

        beam_output_ids = tx.utils.varlength_roll(predictions["sample_id"][:,:,0], -context_len)
        target_ids = tx.utils.varlength_roll(target, -context_len)

        return beam_output_ids, target_ids

    target_ids = tx.utils.varlength_roll(x1x4_ids, -x1_len)


    tau = tf.placeholder(tf.float32, shape=[], name='tau')

    if not FLAGS.sc_rl:
        loss = config_train.w_fine * loss_fine

        loss_dict = {
            'loss': loss,
            'loss_fine': config_train.w_fine * loss_fine,
        }

    else:
        loss = (1 - config_train.w_rl) * config_train.w_fine * loss_fine + config_train.w_rl * (config_train.w_fine_rl * rl_loss_fine) #

        loss_dict = {
            'loss': loss,
            'loss_fine': (1 - config_train.w_rl) * config_train.w_fine * loss_fine,
            'rl_loss_fine': config_train.w_rl * config_train.w_fine_rl * rl_loss_fine,
        }

    ## Inference

    def _infer(context_name):
        helper = tx.modules.TopKSampleEmbeddingHelper(
            embedding=_embedding_fn,
            start_tokens=batch['%s_ids' % context_name][:, 0],
            end_token=end_token,
            top_k=FLAGS.top_k,
            softmax_temperature=FLAGS.temperature)
        outputs_infer, len_infer = decoder(
            context=batch['%s_ids' % context_name],
            context_sequence_length=batch['%s_len' % context_name],
            max_decoding_length=max_decoding_length,
            helper=helper)  # outputs_infer contains sample_id and logits
        yy_ids = tx.utils.varlength_roll(
            outputs_infer.sample_id, -batch['%s_len' % context_name]) # shift beginning indices (context) to end
        yy_len = len_infer - batch['%s_len' % context_name]
        yy_ids = yy_ids[:, :tf.reduce_max(yy_len)]
        return yy_ids, yy_len


    x4_ids_fine, x4_len_fine = _infer('x1')

    def _infer_beam_ids(context_name):
        # beam-search
        predictions = decoder(
            beam_width=5,
            length_penalty=config_train.length_penalty,
            embedding=_embedding_fn,
            context=batch['%s_ids' % context_name],
            context_sequence_length=batch['%s_len' % context_name],
            max_decoding_length=max_decoding_length,
            end_token=end_token,
            mode=tf.estimator.ModeKeys.PREDICT)

        beam_output_ids = tx.utils.varlength_roll(predictions["sample_id"][:, :, 0], -batch['%s_len' % context_name])

        return beam_output_ids
    beam_search_ids = _infer_beam_ids('x1')


    ## Optimization
    trainable_variables = tx.utils.collect_trainable_variables(
        [word_embedder, pos_embedder, decoder])

    global_step = tf.Variable(0, trainable=False)
    opt = tx.core.get_optimizer(
        global_step=global_step,
        hparams=config_train.opt)

    if FLAGS.distributed:
        opt = hvd.DistributedOptimizer(opt)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=None,
        optimizer=opt,
        variables=trainable_variables)


    ## Train/eval/test routine
    saver = tf.train.Saver()
    saver_best = tf.train.Saver(max_to_keep=1)
    dev_best = {
        'loss': 1e8, 'loss_fine': 1e8, 'rl_loss_fine': 1e8, 'best_reward': -1e8, 'bleu':0., 'meteor': 0.} #'best_reward': -1e8


    def _log_losses(losses, step=None):
        loss_str = 'loss: %.4f, loss_fine: %.4f, rl_loss_fine: %.4f' % \
            (losses['loss'], losses['loss_fine'], losses['rl_loss_fine']
             )

        if step is not None:
            loss_str = 'step: %d, %s' % (step, loss_str)

        _log(loss_str)

    def _is_head():
        if not FLAGS.distributed:
            return True
        else:
            return hvd.rank() == 0

    def _train_epoch(sess, initial=False):
        """Trains on the training set, and evaluates on the dev set
        periodically.
        """
        # load train arc label data
        train_arc_file = [i.strip().split() for i in open(os.path.join(config_train.arc_data, "train_mapped.txt"))]

        iterator.restart_dataset(sess, 'train')

        while True:
            try:
                # (1) Get data and yy sample
                fetches_data = {
                    'batch': batch,
                    'batch_size': batch_size,
                }
                feed_dict_data = {
                    iterator.handle: iterator.get_handle(sess, 'train'),
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                rets_data = sess.run(fetches_data, feed_dict_data)


                reward_fetches = {
                    'sample_rl': symbols_rl,
                    'sample_len': len_rl,
                    'greedy_sym': symbols_gr,
                    'greedy_len': len_gr,
                }
                reward_rets = sess.run(reward_fetches, feed_dict={
                    x1_ids: rets_data['batch']['x1_ids'], x1_len: rets_data['batch']['x1_len'],
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT
                })


                # prepare sample stories for classification
                ids_rl, text_rl = _get_text(proc, reward_rets['sample_rl'], reward_rets['sample_len']) #list of list
                story_rl = format_generated_stories_for_clf(text_rl, FLAGS.rl_method)
                #print("Rl Story: ", story_rl)
                _, text_base = _get_text(proc, reward_rets['greedy_sym'], reward_rets['greedy_len'])
                story_base = format_generated_stories_for_clf(text_base, FLAGS.rl_method)
                #print("Greedy Story", story_base)

                # add reward calculation here
                reward_rl = get_reward(predictor, story_rl, rets_data['batch']['unique_id'], train_arc_file , method=FLAGS.rl_method) 
                reward_base = get_reward(predictor, story_base, rets_data['batch']['unique_id'], train_arc_file, method=FLAGS.rl_method)

                # self-critical reward
                reward_sc = [rr - rb for rr, rb in zip(reward_rl, reward_base)] # class list
                # print(reward_rl, reward_base, reward_sc)

                ids_rl = utils.list_strip_eos(ids_rl, end_token)
                new_in_sample_ids, new_in_sample_len = _fix(ids_rl, end_token)

                # (2) Optimize loss
                feed_dict = {
                    x1_ids: rets_data['batch']['x1_ids'],
                    x1_len: rets_data['batch']['x1_len'],
                    x1x4_ids: rets_data['batch']['x1x4_ids'],
                    x1x4_len: rets_data['batch']['x1x4_len'],
                    sampled_story: new_in_sample_ids,
                    sampled_story_len: new_in_sample_len,
                    tau: config_train.tau,
                    tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                    reward: np.array(reward_sc)
                }

                fetches = {
                   'train_op': train_op,
                    'step': global_step,
                }
                fetches.update(loss_dict)

                rets = sess.run(fetches, feed_dict, options=run_opts)
                step = rets['step']

                dis_steps = config_train.display_steps

                if _is_head() and dis_steps > 0 and step % dis_steps == 0:
                    _log_losses(rets, step)

                eval_steps = config_train.eval_steps
                if _is_head() and eval_steps > 0 and step % eval_steps == 0:
                    _dev_epoch(sess, evaluate_func=evaluate_full)
                # not used
                sample_steps = config_train.sample_steps
                if _is_head() and sample_steps > 0 and step % sample_steps == 0:
                    print('-----------testing-----------------')
                    _test_epoch(sess, step=step)
                # not used
                ckpt_steps = config_train.checkpoint_steps
                if _is_head() and ckpt_steps > 0 and step % ckpt_steps == 0:
                    ckpt_fn = os.path.join(output_dir, 'model.ckpt')
                    ckpt_fn = saver.save(sess, ckpt_fn, global_step=step)
                    _log('Checkpoint to {}'.format(ckpt_fn))

            except tf.errors.OutOfRangeError:
                break

    def _dev_epoch(sess, evaluate_func=evaluate_full):
        """Evaluates on the dev set.
        """
        dev_arc_file = [i.strip().split() for i in open(os.path.join(config_train.arc_data, "dev_mapped.txt"))]
        with open(os.path.join(config_train.tfrecord_data_dir, "x4_emo_features.dev"), 'rb') as fp:
            emotion_feats = np.array(pickle.load(fp))
        iterator.restart_dataset(sess, 'dev')

        nsamples = 0
        hypotheses=[]
        references = []
        reward_score = []
        losses = []
        hypotheses_dict = {}

        while True:
            try:

                # (1) Get data and yy sample
                fetches_data = {
                    'batch': batch,
                    'batch_size': batch_size,
                }
                feed_dict_data = {
                    iterator.handle: iterator.get_handle(sess, 'dev'),
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                rets_data = sess.run(fetches_data, feed_dict_data)


                # (2) eval loss
                feed_dict = {
                    x1_ids: rets_data['batch']['x1_ids'],
                    x1_len: rets_data['batch']['x1_len'],
                    x1x4_ids: rets_data['batch']['x1x4_ids'],
                    x1x4_len: rets_data['batch']['x1x4_len'],
                    # x4_emo: rets_data['batch']['x4_emo'],
                    tau: config_train.tau,
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }

                # rets_loss = sess.run(fetches, feed_dict)


                fetches = {
                    'loss_fine': loss_dict['loss_fine'],
                    #'beam_search_ids': beam_search_ids,
                    'greedy_sym': symbols_gr,
                    'greedy_len': len_gr,
                    'target_ids': target_ids
                }
                rets = sess.run(fetches, feed_dict)

                losses.append(rets['loss_fine'])
                _, beam_text = _get_text(proc, rets['greedy_sym'], rets['greedy_len'])
                beam_story = format_generated_stories_for_clf(beam_text, FLAGS.rl_method)
                _, target_text = _get_text(proc, rets['target_ids'], rets_data['batch']['x1x4_len'])

                hypotheses.extend(beam_text)
                references.extend(target_text)
                hypotheses_dict_ = generate_all_valid_sample_dict(predictor, rets_data['batch']['unique_id'], beam_story, method=FLAGS.rl_method)
                for key, react in hypotheses_dict_.items():
                    if key not in hypotheses_dict:
                        hypotheses_dict[key] = react # dictionary key=unique_id value =list of list



                nsamples += rets_data['batch_size']
            except tf.errors.OutOfRangeError:
                break

        avg_loss = np.mean(losses)
        metrics = evaluate_func(references, hypotheses, hypotheses_dict, dev_arc_file, emotion_feats, method=FLAGS.rl_method)
        msg = 'loss_fine: %.4f, bleu: %.4f, meteor: %.4f, reward: %.4f' % \
            (avg_loss, metrics['bleu'], metrics['meteor'], metrics["best_reward"]
             )

        _log('nsamples validation: %d' % nsamples)
        _log(msg)


        if FLAGS.best_model == "emotion":
            if FLAGS.do_train and metrics["best_reward"] > dev_best['best_reward']:
                # dev_best.update(results.avg())
                dev_best['loss_fine'] = avg_loss
                dev_best['best_reward'] = metrics["best_reward"]
                dev_best.update(metrics)
                ckpt_fn = os.path.join(output_dir, 'model_best.ckpt')
                ckpt_fn = saver_best.save(sess, ckpt_fn)
                _log('Checkpoint best to {}'.format(ckpt_fn))

        elif FLAGS.best_model == "bleu":
            if FLAGS.do_train and metrics["bleu"] > dev_best['bleu']:
                # dev_best.update(results.avg())
                dev_best['loss_fine'] = avg_loss
                dev_best['best_reward'] = metrics["best_reward"]
                dev_best.update(metrics)
                ckpt_fn = os.path.join(output_dir, 'model_best.ckpt')
                ckpt_fn = saver_best.save(sess, ckpt_fn)
                _log('Checkpoint best to {}'.format(ckpt_fn))

        elif FLAGS.do_train and avg_loss < dev_best['loss']:
            # dev_best.update(results.avg())
            dev_best['loss_fine'] = avg_loss
            dev_best.update(metrics)
            dev_best['best_reward'] = metrics["best_reward"]
            ckpt_fn = os.path.join(output_dir, 'model_best.ckpt')
            ckpt_fn = saver_best.save(sess, ckpt_fn)
            _log('Checkpoint best to {}'.format(ckpt_fn))

    def _test_epoch(sess, step=None):
        """Generates samples on the test set.
        """
        iterator.restart_dataset(sess, 'test')

        _all_inputs = []
        _all_samples = []

        if FLAGS.finetune:
            _log('Generation input: x1')
            fetches = {
                'inputs': batch['x1_ids'],
                'length': batch['x1_len'],
                'samples_length': x4_len_fine,
                'samples': x4_ids_fine
            }
            res_fn_appendix = "x1"



        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'test'),
                    tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                rets = sess.run(fetches, feed_dict=feed_dict)

                _inputs = []
                for i, l in zip(rets['inputs'], rets['length']):
                    # Delete padding
                    _inputs.append(i[:l].tolist())
                _all_inputs.extend(_inputs)

                _samples = []
                for s, l in zip(rets['samples'], rets['samples_length']):
                    _samples.append(s[:l].tolist()) # rets['samples'] are np array [bs, max_seq_len=200]

                _all_samples.extend(_samples)

            except tf.errors.OutOfRangeError:
                break

        # Parse samples and write to file

        eos_token_id = proc.encoder['<|endoftext|>']

        _all_input_text = []
        for i in _all_inputs:
            if i[0] == eos_token_id:
                i = i[1:]
            i_text = proc.decode(i)
            _all_input_text.append(i_text)
        _all_input_text = tx.utils.strip_eos(_all_input_text,
                                             eos_token='<|endoftext|>')

        _all_samples_text = []
        for i, s in zip(_all_inputs, _all_samples):
            s_text = proc.decode(s)
            s_text = s_text.strip(" |").replace('\n', ' ')
            _all_samples_text.append(s_text)
        _all_samples_text = tx.utils.strip_eos(_all_samples_text,
                                             eos_token='<|endoftext|>')

        if step is None:
            fn = "test_samples_%s.tsv" % res_fn_appendix
        else:
            fn = "test_samples_%s_%d.tsv" % (res_fn_appendix, step)
        output_file = os.path.join(output_dir, fn)
        _log('Write samples to {}'.format(output_file))
        tx.utils.write_paired_text(
            _all_input_text, _all_samples_text, output_file)


    # Broadcasts global variables from rank-0 process
    if FLAGS.distributed:
        bcast = hvd.broadcast_global_variables(0)

    session_config = tf.ConfigProto()
    if FLAGS.distributed:
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())
        session_config.gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        #smry_writer = tf.summary.FileWriter(FLAGS.output_dir, graph=sess.graph)

        if FLAGS.distributed:
            bcast.run()

        #Restores trained model if specified
        if FLAGS.checkpoint:
           _log('Restore from {}'.format(FLAGS.checkpoint))
           saver.restore(sess, FLAGS.checkpoint)
        elif FLAGS.pretrain_checkpoint:
           _log('Restore from {}'.format(FLAGS.pretrain_checkpoint))
           model_utils.init_gpt2_checkpoint(sess, FLAGS.pretrain_checkpoint)
           print("\nFinished loading\n")
           saver.save(sess, output_dir + '/gpt2_model.ckpt')


        iterator.initialize_dataset(sess)

        if FLAGS.do_train:
            for epoch in range(config_train.max_train_epoch):
                print("Training epoch {}".format(epoch))
                _train_epoch(sess, epoch==0)
            saver.save(sess, output_dir + '/model.ckpt')

        if FLAGS.do_eval:
            _dev_epoch(sess)

        if FLAGS.do_test:
            _test_epoch(sess)


if __name__ == "__main__":
    tf.app.run()


