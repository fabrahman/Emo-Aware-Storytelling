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

import os
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from data_utils import model_utils, processor, utils

# pylint: disable=invalid-name, too-many-locals, too-many-statements, no-member
# pylint: disable=invalid-name, too-many-locals, too-many-statements, no-member
# pylint: disable=too-many-branches


# Only finetune on ROC

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
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_test", False, "Whether to run test on the test set.")
flags.DEFINE_bool("distributed", False, "Whether to run in distributed mode.")
flags.DEFINE_bool("finetune", False, "Whether to test on finetune mode.")


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

    # For training
    def _get_recon_loss(ids, full_len, prefix_len, mask_prefix=True, do_print=False):
        ids = ids[:,:tf.reduce_max(full_len)]
        batch_size__ = tf.shape(ids)[0]
        seq_len = tf.fill([batch_size__], tf.shape(ids)[1])
        pos_embeds = pos_embedder(sequence_length=seq_len)
        input_embeds = word_embedder(ids) + pos_embeds

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
                average_across_batch=True,
                sum_over_batch=False)

        return loss_recon


    ## ROC Loss-1: fine-tune loss
    x1_len = tf.placeholder(tf.int32, shape=[None], name='x1_len')
    x1x4_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1x4_ids')
    x1x4_len = tf.placeholder(tf.int32, shape=[None], name='x1x4_len')

    loss_fine = _get_recon_loss(x1x4_ids, x1x4_len, x1_len)

    tau = tf.placeholder(tf.float32, shape=[], name='tau')

    # generate soft yy
    def _soft_embedding_fn(soft_ids, times):
        return word_embedder(soft_ids=soft_ids) + pos_embedder(times)
    end_token = proc.encoder['<|endoftext|>']

    if not FLAGS.supervised:
        loss = config_train.w_fine * loss_fine

        loss_dict = {
            'loss': loss,
            'loss_fine': config_train.w_fine * loss_fine,
        }
    else:
        loss = loss_yy

        loss_dict = {
            'loss': loss,
            'loss_yy': loss_yy,
            # dumb
            'loss_mask_recon': tf.constant(0),
            'loss_bt': tf.constant(0),
            'loss_d_xx2': tf.constant(0),
            'loss_d_x2': tf.constant(0),
            'loss_fine': tf.constant(0),
            'loss_xx2': tf.constant(0)
        }

    ## Inference
    def _embedding_fn(ids, times):
        return word_embedder(ids) + pos_embedder(times)

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
            helper=helper)
        yy_ids = tx.utils.varlength_roll(
            outputs_infer.sample_id, -batch['%s_len' % context_name])
        yy_len = len_infer - batch['%s_len' % context_name]
        yy_ids = yy_ids[:, :tf.reduce_max(yy_len)]
        # yy_logits = outputs_infer.logits
        # # yy_loss = _evaluate_loss_test(yy_logits, target_name, context_name)

        return yy_ids, yy_len

    def _evaluate_loss_test(target_name, context_name, bpe_loss=FLAGS.bpe_loss):
        ids = batch['%s_ids' % target_name]
        full_len = batch['%s_len' % target_name]
        ids = ids[:, :tf.reduce_max(full_len)]

        batch_size__ = tf.shape(ids)[0]
        seq_len = tf.fill([batch_size__], tf.shape(ids)[1])
        pos_embeds = pos_embedder(sequence_length=seq_len)
        input_embeds = word_embedder(ids) + pos_embeds

        # greedy output
        outputs = decoder(inputs=input_embeds, decoding_strategy='train_greedy')
        max_full_len = tf.reduce_max(full_len)
        logits = outputs.logits[:, :max_full_len]

        test_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=ids[:, 1:],
            logits=logits[:, :-1, :],
            sequence_length=full_len - 1,
            average_across_timesteps=False,
            sum_over_timesteps=False, # not bpe_loss, # True,
            average_across_batch=False,
            sum_over_batch=False)
        mask_recon = tf.sequence_mask(
            full_len - 1,
            dtype=tf.float32)
        mask_recon_prefix = 1 - tf.sequence_mask(
            batch['%s_len' % context_name] - 1,
            maxlen=max_full_len - 1,  # max_decoding_length-1,
            dtype=tf.float32)
        mask_recon = mask_recon * mask_recon_prefix

        test_loss = tx.utils.reduce_with_weights(
            tensor=test_loss,
            weights=mask_recon,
            average_across_batch=bpe_loss,
            average_across_remaining=bpe_loss,
            sum_over_remaining=not bpe_loss)

        return test_loss # [bs,] ?



    x4_ids_fine, x4_len_fine = _infer('x1')
    x4_loss_fine = _evaluate_loss_test('x1x4', 'x1')

    ## Optimization

    def _get_beam_ids(context_name):
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
    beam_search_ids = _get_beam_ids('x1')

    def _get_greedy_story(context_name):

        greedy_res, greedy_len = decoder(
            decoding_strategy='infer_greedy',
            embedding=_embedding_fn,
            context=batch['%s_ids' % context_name],
            context_sequence_length=batch['%s_len' % context_name],
            max_decoding_length=max_decoding_length,
            end_token=end_token,
            mode=tf.estimator.ModeKeys.PREDICT)

        greedy_ids = tx.utils.varlength_roll(greedy_res.sample_id, -batch['%s_len' % context_name])
        greedy_ids_len = greedy_len - batch['%s_len' % context_name]
        greedy_ids = greedy_ids[:, :tf.reduce_max(greedy_ids_len)]

        return greedy_ids, greedy_ids_len
    greedy_ids, greedy_len = _get_greedy_story('x1')





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
        'loss': 1e8, 'loss_fine': 1e8}


    def _log_losses(losses, step=None):
        loss_str = 'loss: %.4f, loss_fine: %.4f' % \
            (losses['loss'], losses['loss_fine'])

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


                # (2) Optimize loss
                feed_dict = {
                    #x1_ids: rets_data['batch']['x1_ids'],
                    x1_len: rets_data['batch']['x1_len'],
                    x1x4_ids: rets_data['batch']['x1x4_ids'],
                    x1x4_len: rets_data['batch']['x1x4_len'],
                    tau: config_train.tau,
                    tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                }

                fetches = {
                    'train_op': train_op,
                    'step': global_step,
                }
                fetches.update(loss_dict)

                rets = sess.run(fetches, feed_dict)
                step = rets['step']

                dis_steps = config_train.display_steps

                if _is_head() and dis_steps > 0 and step % dis_steps == 0:
                    _log_losses(rets, step)

                eval_steps = config_train.eval_steps
                if _is_head() and eval_steps > 0 and step % eval_steps == 0:
                    _dev_epoch(sess)
                sample_steps = config_train.sample_steps
                if _is_head() and sample_steps > 0 and step % sample_steps == 0:
                    print('-----------testing-----------------')
                    _test_epoch(sess, step=step)

                ckpt_steps = config_train.checkpoint_steps
                if _is_head() and ckpt_steps > 0 and step % ckpt_steps == 0:
                    ckpt_fn = os.path.join(output_dir, 'model.ckpt')
                    ckpt_fn = saver.save(sess, ckpt_fn, global_step=step)
                    _log('Checkpoint to {}'.format(ckpt_fn))

            except tf.errors.OutOfRangeError:
                break

    def _dev_epoch(sess):
        """Evaluates on the dev set.
        """
        iterator.restart_dataset(sess, 'dev')

        results = tx.utils.AverageRecorder()
        nsamples = 0
        fetches = {}
        fetches.update(loss_dict)
        # i = 0

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
                    #x1_ids: rets_data['batch']['x1_ids'],
                    x1_len: rets_data['batch']['x1_len'],
                    x1x4_ids: rets_data['batch']['x1x4_ids'],
                    x1x4_len: rets_data['batch']['x1x4_len'],
                    tau: config_train.tau,
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }

                rets = sess.run(fetches, feed_dict)

                results.add(rets, weight=rets_data['batch_size'])
                nsamples += rets_data['batch_size']
            except tf.errors.OutOfRangeError:
                break

        _log_losses(results.avg())
        _log('nsamples: %d' % nsamples)

        avg_loss = results.avg('loss')
        if FLAGS.do_train and avg_loss < dev_best['loss']:
            dev_best.update(results.avg())
            ckpt_fn = os.path.join(output_dir, 'model_best.ckpt')
            ckpt_fn = saver_best.save(sess, ckpt_fn)
            _log('Checkpoint best to {}'.format(ckpt_fn))

    def _test_epoch(sess, step=None):
        """Generates samples on the test set.
        """
        iterator.restart_dataset(sess, 'test')

        _all_inputs = []
        _all_samples = []
        _all_loss = []

        # if FLAGS.finetune and FLAGS.roc:
        #     raise ValueError('Cannot set --finetune and --roc at the same time')
        if FLAGS.finetune:
            _log('Generation input: x1')
            if FLAGS.greedy:
                fetches = {
                    'inputs': batch['x1_ids'],
                    'length': batch['x1_len'],
                    'samples_length': greedy_len,
                    'samples': greedy_ids
                }
            elif FLAGS.beam:
                fetches = {
                    'inputs': batch['x1_ids'],
                    'length': batch['x1_len'],
                    # 'samples_length': x4_len_fine,
                    'samples': beam_search_ids
                }
            else:
                fetches = {
                    'inputs': batch['x1_ids'],
                    'length': batch['x1_len'],
                    'samples_length': x4_len_fine,
                    'samples': x4_ids_fine,
                    'sample_loss': x4_loss_fine,
                    'outputs': batch['x1x4_ids'],
                    'out_length': batch['x1x4_len']
                }
            res_fn_appendix = "x1"




        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'test'),
                    tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                rets = sess.run(fetches, feed_dict=feed_dict)

                # ! ----
                _inputs = []
                for i, l in zip(rets['inputs'], rets['length']):
                    # Delete padding
                    _inputs.append(i[:l].tolist())
                _all_inputs.extend(_inputs)

                _samples = []

                if not FLAGS.beam:
                    for s, l in zip(rets['samples'], rets['samples_length']):
                        _samples.append(s[:l].tolist())

                else:
                    _samples.extend(h.tolist() for h in rets['samples'])
                    _samples = utils.list_strip_eos(_samples, eos_token=proc.encoder['<|endoftext|>'])
                _all_samples.extend(_samples)
                # ----!

                _loss = []
                if not FLAGS.bpe_loss:
                    for los in rets["sample_loss"]:
                        _loss.append(los)
                else:
                    _loss = [rets["sample_loss"]]

                _all_loss.extend(_loss)


            except tf.errors.OutOfRangeError:
                break

        # Parse samples and write to file

        eos_token_id = proc.encoder['<|endoftext|>']

        # !----
        _all_input_text = []
        for i in _all_inputs:
            if i[0] == eos_token_id:
                i = i[1:]
            i_text = proc.decode(i)
            _all_input_text.append(i_text)
        _all_input_text = tx.utils.strip_eos(_all_input_text,
                                             eos_token='<|endoftext|>')

        _all_samples_text = []
        for j, (i, s) in enumerate(zip(_all_inputs, _all_samples)):
            s_text = proc.decode(s)
            s_text = s_text.replace('\n', ' ')
            # print(s_text)
            _all_samples_text.append(s_text)
            if j % 1000 == 0:
                print("{} stories is process of total {}".format(j, len(_all_inputs)))

        _all_samples_text = tx.utils.strip_eos(_all_samples_text,
                                             eos_token='<|endoftext|>')

        if step is None:
            fn = "test_samples_%s_sample_k%d.tsv" % (res_fn_appendix, FLAGS.top_k)
        else:
            fn = "test_samples_%s_%d_beam.tsv" % (res_fn_appendix, step)
        output_file = os.path.join(output_dir, fn)
        _log('Write samples to {}'.format(output_file))
        if not FLAGS.beam:
            tx.utils.write_paired_text(
            _all_input_text, _all_samples_text, output_file)
            with open(output_file[:-4]+".txt", 'w') as f:
                for item in _all_samples_text:
                    f.write("%s\n" % item.strip(" | "))
        else:
            with open(output_file, 'w') as f:
                for item in _all_samples_text:
                    f.write("%s\n" % item)
        # ----!

        if FLAGS.ppl:
            if not FLAGS.bpe_loss:
                # load target file
                target = [i.strip().split() for i in open("emotion_evaluation/baselines/ground-truth/ground_truth_story-processed.txt")]
                for j, (txt, los) in enumerate(zip(target, _all_loss)):
                    _all_loss[j] = los/len(txt)

                np.save(os.path.join(output_dir, "test_loss_word.npy"), np.array(_all_loss))
                avg_loss = np.mean(np.array(_all_loss))
                ppl = np.exp(avg_loss)
                msg = 'test_loss (per word): %.4f, test_perplexity: %.4f' % \
                    (avg_loss, ppl
                     )
            else:
                avg_loss = np.mean(np.array(_all_loss))
                ppl = np.exp(avg_loss)
                msg = 'test_loss (bpe): %.4f, test_perplexity: %.4f' % \
                    (avg_loss, ppl
                     )

            _log(msg)


    # Broadcasts global variables from rank-0 process
    if FLAGS.distributed:
        bcast = hvd.broadcast_global_variables(0)

    session_config = tf.ConfigProto()
    if FLAGS.distributed:
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

#        smry_writer = tf.summary.FileWriter(FLAGS.output_dir, graph=sess.graph)

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


