
#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import math
import time
import json
import random

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from helper_scripts.data_iterator import TrainIterator

from word_embed_attention import Seq2SeqModel
from tqdm import tqdm


# Context pre-processing
tf.app.flags.DEFINE_string('user_vocab_file', 'vocab_data/user_first_response_jan_2018_p1_05_06_2018.p', 'Path to encoder vocabulary')
tf.app.flags.DEFINE_string('agent_vocab_file', 'vocab_data/agent_first_response_jan_2018_p1_05_06_2018.p', 'Path to decoder vocabulary')

# Dataset
tf.app.flags.DEFINE_string('train_data', 'data/first-response-jan-2018-processed-p1.txt', 'Path to training data')
tf.app.flags.DEFINE_string('valid_data', 'data/first-response-jan-2018-processed-p2.txt', 'Path to validating data')
tf.app.flags.DEFINE_string('delimiter', ' +++$+++ ', 'delimiter seprating the source and target fields')

# Network parameters
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_units', 300, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('depth', 1, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 150, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('num_encoder_symbols', 4201, 'Encoder vocabulary size')
tf.app.flags.DEFINE_integer('num_decoder_symbols', 3591, 'Decoder vocabulary size')

tf.app.flags.DEFINE_boolean('use_residual', False, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')
# tf.app.flags.DEFINE_integer('beam_width', 1, 'beam width dimension')
# tf.app.flags.DEFINE_integer('max_decode_step', 25, 'Not Sure')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 600, 'Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 80, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_seq_length_enc', 53, 'Encoder Maximum sequence length')
tf.app.flags.DEFINE_integer('max_seq_length_dec', 61, 'Decoder Maximum sequence length')
tf.app.flags.DEFINE_integer('display_freq', 186, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 1860, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 186, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', 'model_05_06_2018/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'optus_one_turn_chat_bot.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', False, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')


FLAGS = tf.app.flags.FLAGS

def create_model(session, FLAGS):

    config = OrderedDict(sorted(FLAGS.__flags.items()))
    model = Seq2SeqModel(config, 'train')

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print('Created new model parameters..')
        session.run(tf.global_variables_initializer())

    return model

def train():
    # Load parallel data to train
    print('Loading training data..')
    train_set = TrainIterator(FLAGS.train_data,
                              FLAGS.user_vocab_file,
                              FLAGS.agent_vocab_file,
                              batch_size=FLAGS.batch_size,
                              maxlen_enc=FLAGS.max_seq_length_enc,
                              maxlen_dec=FLAGS.max_seq_length_dec,
                              shuffle_each_epoch=FLAGS.shuffle_each_epoch,
                              delimiter=FLAGS.delimiter,)

    # if FLAGS.source_valid_data and FLAGS.target_valid_data:
    if FLAGS.valid_data:
        print('Loading validation data..')
        valid_set = TrainIterator(FLAGS.valid_data,
                              FLAGS.user_vocab_file,
                              FLAGS.agent_vocab_file,
                              batch_size=FLAGS.batch_size,
                              maxlen_enc=FLAGS.max_seq_length_enc,
                              maxlen_dec=FLAGS.max_seq_length_dec,
                              shuffle_each_epoch=FLAGS.shuffle_each_epoch,
                              delimiter=FLAGS.delimiter,)

    else:
        valid_set = None

    # Initiate TF session
    with tf.Session() as sess:

        # Create a new model or reload existing checkpoint
        model = create_model(sess, FLAGS)

        # Create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()

        # Training loop
        print('Training..')
        num_batches_per_epoch = 111070/FLAGS.batch_size
        for epoch_idx in tqdm(range(FLAGS.max_epochs)):
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print('Training is already complete.', \
                      'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs))
                break

            # for batch_train_data in tqdm(train_set, total=num_batches_per_epoch):
            for batch_train_data in train_set:
                encoder_inputs = np.array(batch_train_data[0])
                encoder_inputs_length = np.array(batch_train_data[1])
                decoder_inputs = np.array(batch_train_data[2])
                decoder_inputs_length = np.array(batch_train_data[3])
                decoder_targets = np.array(batch_train_data[4])

                tmp_enc = sum([1 for x in encoder_inputs_length if x<=0])
                if tmp_enc>0:
                    print(chat_level_batch)
                    raise ValueError("Train data has empty Encoder sequences ... ")

                tmp_dec = sum([1 for x in decoder_inputs_length if x<=0])
                if tmp_dec>0:
                    raise ValueError("Train data has empty Decoder sequences ... ")

                # Execute a single training step
                step_loss, summary = model.train(sess, encoder_inputs,
                                                encoder_inputs_length,
                                                decoder_inputs,
                                                decoder_inputs_length,
                                                decoder_targets,)

                loss += float(step_loss) / FLAGS.display_freq

                if model.global_step.eval() % FLAGS.display_freq == 0:

                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / FLAGS.display_freq

                    print('Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(), 'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time ', step_time)

                    loss = 0
                    start_time = time.time()

                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # Execute a validation step
                if valid_set and model.global_step.eval() % FLAGS.valid_freq == 0 and epoch_idx>19 and epoch_idx % 5==0:
                    print('Validation step')
                    valid_loss = 0.0
                    valid_sents_seen = 0

                    for batch_indx, batch_valid_data in enumerate(valid_set):
                        encoder_inputs = np.array(batch_valid_data[0])
                        encoder_inputs_length = np.array(batch_valid_data[1])
                        decoder_inputs = np.array(batch_valid_data[2])
                        decoder_inputs_length = np.array(batch_valid_data[3])
                        decoder_targets = np.array(batch_valid_data[4])

                        # Compute validation loss: average per word cross entropy loss
                        step_loss, summary = model.eval(sess, encoder_inputs,
                                                encoder_inputs_length,
                                                decoder_inputs,
                                                decoder_inputs_length,
                                                decoder_targets,)


                        valid_loss += step_loss


                    valid_loss = valid_loss / (batch_indx+1)
                    print('Valid perplexity: {0:.2f}'.format(math.exp(valid_loss)))

                # Save the model checkpoint
                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print('Saving the model..')
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    json.dump(model.config,
                              open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'wb'),
                              indent=2)


            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            print('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))

        print('Saving the last model..')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'wb'),
                  indent=2)

    print('Training Terminated')



def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
    train()
