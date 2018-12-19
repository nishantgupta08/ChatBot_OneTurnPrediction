
#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
import json
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from helper_scripts import util
from helper_scripts.data_iterator import CLIDecodeProcessor

from seq2seq_with_attention import Seq2SeqModel

# Decoding parameters
tf.app.flags.DEFINE_string('user_vocab_file', 'vocab_data/user_first_response_jan_2018_p1_05_06_2018.p', 'Path to encoder vocabulary')
tf.app.flags.DEFINE_string('agent_vocab_file', 'vocab_data/agent_first_response_jan_2018_p1_05_06_2018.p', 'Path to decoder vocabulary')
# tf.app.flags.DEFINE_string('delimiter', ' +++$+++ ', 'delimiter seprating the source and target fields')
tf.app.flags.DEFINE_integer('beam_width', 1, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('max_decode_step', 61, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_string('model_path', 'model_07_06_2018/optus_one_turn_chat_bot.ckpt-3720', 'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_integer('max_seq_length', 61, 'Maximum sequence length')  #TODO ..get the value from train parameters

FLAGS = tf.app.flags.FLAGS

def load_config(FLAGS):

    config = util.unicode_to_utf8(
        json.load(open('%s.json' % FLAGS.model_path, 'rb')))
        # json.load(open('model/translate.ckpt-120.json', 'rb')))
    for key, value in list(FLAGS.__flags.items()):
        config[key] = value

    return config


def load_model(session, config):

    model = Seq2SeqModel(config, 'decode')
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print('Reloading model parameters..')
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


def decode():
    # Load model config
    config = load_config(FLAGS)

    # Load source data to decode
    decode_obj = CLIDecodeProcessor(FLAGS.user_vocab_file, FLAGS.agent_vocab_file, FLAGS.max_seq_length)

    # Initiate TF session
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
        # log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    with tf.Session() as sess:

        model = load_model(sess, config)

        hist_user_text = ""
        hist_agent_text = ""

        # _, _, id_2_word_dict = pickle.load(open(FLAGS.vocab_file_name, 'r'))

        while True:
            # print "\nEnter Input ..."
            print("\n[Input]: ", end=' ')
            vis_input = str(input()).lower().strip()
            seq_cmd_line, len_cmd_line = decode_obj.get_cli_decode_data(vis_input)

            predicted_ids = model.predict(sess, encoder_inputs=seq_cmd_line,
                                           encoder_inputs_length=len_cmd_line,)

            # print "Predicted IDs: ", predicted_ids[0]
            # import pdb; pdb.set_trace()
            pred_decode = decode_obj.seq2words(predicted_ids[0].tolist()[0])
            print("[Prediction]: ", pred_decode)



def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run()

