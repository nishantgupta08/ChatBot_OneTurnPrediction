
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

# from tensorflow.contrib.rnn import GRUCell
# from tensorflow.contrib.rnn import LSTMCell
# from tensorflow.contrib.rnn import MultiRNNCell
# from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

class Seq2SeqModel(object):

    def __init__(self, config, mode):

        assert mode.lower() in ['train', 'decode']

        self.config = config
        self.mode = mode.lower()

        self.cell_type = config['cell_type'].value
        self.hidden_units = config['hidden_units'].value
        self.depth = config['depth'].value
        self.attention_type = config['attention_type'].value
        self.embedding_size = config['embedding_size'].value

        self.num_encoder_symbols = config['num_encoder_symbols'].value
        self.num_decoder_symbols = config['num_decoder_symbols'].value

        self.use_residual = config['use_residual'].value
        self.attn_input_feeding = config['attn_input_feeding'].value
        self.use_dropout = config['use_dropout'].value
        self.keep_prob = 1.0 - config['dropout_rate'].value

        self.optimizer = config['optimizer'].value
        self.learning_rate = config['learning_rate'].value
        self.max_gradient_norm = config['max_gradient_norm'].value
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
	    tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.dtype = tf.float16 if config['use_fp16'] else tf.float32
        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')

        # self.max_seq_length = config['max_seq_length']

        self.use_beamsearch_decode=False
        if self.mode == 'decode':
            self.beam_width = config['beam_width']
            self.use_beamsearch_decode = True if self.beam_width > 1 else False
            self.max_decode_step = config['max_decode_step']

        self.PAD_ID = 0
        self.GO_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3

        self.build_model()

    def build_model(self):
        print("building model..")

        # Building encoder and decoder networks
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()


    def init_placeholders(self):
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
            shape=(None, None), name='encoder_inputs')

        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='encoder_inputs_length')


        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        if self.mode == 'train':

            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name='decoder_inputs')
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='decoder_inputs_length')
            # decoder_targets: [batch_size, max_time_steps]
            self.decoder_targets = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name='decoder_targets')


    def build_encoder(self):
        print("building encoder..")
        with tf.variable_scope('encoder'):
            # Building encoder_cell
            # self.encoder_cell = self.build_encoder_cell()
            self.encoder_cell = self.build_single_cell()

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.encoder_embeddings = tf.get_variable(name='embedding',
                shape=[self.num_encoder_symbols, self.embedding_size],
                initializer=initializer, dtype=self.dtype)

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=self.encoder_inputs)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded2 = input_layer(self.encoder_inputs_embedded)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            _, self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded2,
                sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                time_major=False)


    def build_decoder(self):
        print("building decoder and attention..")
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            # self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.decoder_embeddings = tf.get_variable(name='embedding',
                shape=[self.num_decoder_symbols, self.embedding_size],
                initializer=initializer, dtype=self.dtype)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.num_decoder_symbols, name='output_projection_layer')

            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings, ids=self.decoder_inputs)

                # Embedded inputs having gone through input projection layer
                self.decoder_inputs_embedded2 = input_layer(self.decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded2,
                                                   sequence_length=self.decoder_inputs_length,
                                                   time_major=False,
                                                   name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=self.decoder_initial_state,
                                                   output_layer=output_layer)


                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length)

                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                (self.decoder_outputs_train, self.decoder_last_state_train,
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length))

                # logits_train: [batch_size, max_time_step + 1, num_hidden_units]
                self.decoder_hidden_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)


                masks = tf.sequence_mask(lengths=self.decoder_inputs_length,
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                self.loss = seq2seq.sequence_loss(logits=self.decoder_hidden_logits_train,
                                                  targets=self.decoder_targets,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True,)
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)

                # Contruct graphs for minimizing loss
                self.init_optimizer()

            elif self.mode == 'decode':

                start_tokens = tf.ones([self.batch_size,], tf.int32) * self.GO_ID
                end_token = self.EOS_ID

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=end_token,
                                                                    embedding=embed_and_input_proj)

                    # Basic decoder performs greedy decoding at each time step
                    print("building greedy decoder..")

                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                   helper=decoding_helper,
                                                   initial_state=self.decoder_initial_state,
                                                   output_layer=output_layer)


                else:
                    # Beamsearch is used to approximately find the most likely translation
                    print("building beamsearch decoder..")
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                               embedding=embed_and_input_proj,
                                                               start_tokens=start_tokens,
                                                               end_token=end_token,
                                                               initial_state=self.decoder_initial_state,
                                                               beam_width=self.beam_width,
                                                               output_layer=output_layer,)

                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    #impute_finished=True,	# error occurs
                    maximum_iterations=self.max_decode_step))

                if not self.use_beamsearch_decode:

                    self.decoder_hidden_logits_decode = tf.identity(self.decoder_outputs_decode.rnn_output)

                    # Use argmax to extract decoder symbols to emit
                    self.decoder_pred_decode = tf.argmax(self.decoder_hidden_logits_decode, axis=-1,
                                                    name='decoder_pred_decode')


                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids


    def build_single_cell(self):
        cell_type = LSTMCell
        if (self.cell_type.lower() == 'gru'):
            cell_type = GRUCell
        cell = cell_type(self.hidden_units)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder,)
        if self.use_residual:
            cell = ResidualWrapper(cell)

        return cell


    # Building encoder cell
    def build_encoder_cell (self):

        return MultiRNNCell([self.build_single_cell() for i in range(self.depth)])


    # Building decoder cell and attention. Also returns decoder_initial_state
    def build_decoder_cell(self):

        decoder_initial_state = self.encoder_last_state

        decoder_cell = self.build_single_cell()

        return decoder_cell, decoder_initial_state


    def init_optimizer(self):
        print("setting optimizer..")
        # Gradients and SGD update operation for training the model
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        # print "Trainable Params: ", trainable_params
        gradients = tf.gradients(self.loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
        self.updates = self.opt.apply_gradients(
            list(zip(clip_gradients, trainable_params)), global_step=self.global_step)

    def save(self, sess, path, var_list=None, global_step=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)

        # temporary code
        #del tf.get_collection_ref('LAYER_NAME_UIDS')[0]
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print(('model saved at %s' % save_path))


    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print(('model restored from %s' % path))


    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length, decoder_targets):
        """Run a train step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length,
                                      decoder_targets, decode=False,
                                      )
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        output_feed = [self.updates,	# Update Op that does optimization
                       self.loss,	# Loss for current batch
                       self.summary_op]	# Training summary


        outputs = sess.run(output_feed, input_feed)

        return outputs[1], outputs[2]	# loss, summary


    def eval(self, sess, encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length, decoder_targets,):
        """Run a evaluation step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length,
                                      decoder_targets, decode=False,
                                      )
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.loss,	# Loss for current batch
                       self.summary_op]	# Training summary
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]	# loss


    def predict(self, sess, encoder_inputs, encoder_inputs_length):

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs=None, decoder_inputs_length=None,
                                      decoder_targets=None, decode=True)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.decoder_pred_decode]
        outputs = sess.run(output_feed, input_feed)

        return outputs	# BeamSearchDecoder: [batch_size, max_time_step, beam_width]


    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decoder_targets, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                    "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                    "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length
            input_feed[self.decoder_targets.name] = decoder_targets

        return input_feed

