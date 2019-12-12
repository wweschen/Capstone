from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import tensorflow as tf
import itertools

from absl import logging

from modeling import tf_utils
from bert import bert_modeling


def get_initializer(initializer_range=0.02):
  """Creates a `tf.initializers.truncated_normal` with the given range.

  Args:
    initializer_range: float, initializer range for stddev.

  Returns:
    TruncatedNormal initializer with stddev = `initializer_range`.
  """
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
    """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
    """
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists
    self.p_gens = p_gens
    self.coverage = coverage

  def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist],
                      p_gens = self.p_gens + [p_gen],
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)

class PGNetConfig(object):
  """Configuration for `PGNetSummaryModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=256,
               emb_dim=128,
               max_enc_steps=400,
               max_dec_steps=100,
               beam_size=4,
               min_dec_steps= 1,
               use_pointer_gen=True,
               use_coverage=True ):


    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.emb_dim = emb_dim
    self.max_enc_steps = max_enc_steps
    self.beam_size = beam_size
    self.max_dec_steps = max_dec_steps
    self.min_dec_steps = min_dec_steps
    self.use_pointer_gen = use_pointer_gen
    self.use_coverage = use_coverage

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = PGNetConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class PGNetDecoderModel(tf.keras.layers.Layer):
    def __init__(self,
                 config,
                 float_type=tf.float32,
                 **kwargs):
        super(PGNetDecoderModel, self).__init__(**kwargs)

        self.config = config

        self.float_type = float_type

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""

        self.embedding_lookup = EmbeddingLookup(self.config.vocab_size,
                                                self.config.hidden_size,
                                                name="word_embeddings")

        self.decoder = AttentionDecoder(self.config.hidden_size, self.config.max_seq_length,
                                        self.config.max_answer_length, get_initializer(),
                                        name="attention_decoder")
        self.output_projector = OutputProjectionLayer(self.config.hidden_size, self.config.vocab_size,
                                        name="output_projector")
        self.final_distribution = FinalDistributionLayer(self.config.hidden_size, self.config.vocab_size,
                                                         name = "final_distribution")

        super(PGNetDecoderModel, self).build(unused_input_shapes)

    def __call__(self,
                 enc_states,
                 answer_ids,
                 answer_mask,
                 **kwargs):
        inputs = (enc_states, answer_ids, answer_mask)
        return super(PGNetDecoderModel, self).__call__(inputs, **kwargs)

    def call(self, inputs ):

        enc_states = inputs[0]
        answer_ids = inputs[1]
        dec_mask = inputs[2]
        batch_size = tf_utils.get_shape_list(answer_ids)[0]
        emb_dec_inputs = [self.embedding_lookup(x) for x in tf.unstack(answer_ids,
                                                                       axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)

        dec_initial_state=tf.zeros([batch_size, self.config.hidden_size])

        #prev_coverage = self.prev_coverage if self.config.mode == "decode" and self.config.use_coverage else None

        decoder_outputs,  dec_out_state,  attn_dists,  p_gens,  coverage = self.decoder(
            emb_dec_inputs,
            dec_initial_state,
            enc_states,
            dec_mask )

        vocab_dists = None #self.output_projector(decoder_outputs)

        if self.config.use_pointer_gen:
            final_dists = self.final_distribution(vocab_dists,  attn_dists,p_gens)
        else:  # final distribution is just vocabulary distribution
            final_dists = vocab_dists

        return final_dists, attn_dists

    def get_config(self):
        config = {"config": self.config.to_dict()}
        base_config = super(PGNetDecoderModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class OneStepDecodeLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 vector_size,
                 attention_length,
                 initial_state_attention=False,
                 pointer_gen=True,
                 use_coverage=False,
                 initializer=None,
                 float_type=tf.float32,
                 **kwargs):
        super(OneStepDecodeLayer, self).__init__(**kwargs)
        self.initializer = initializer
        self.float_type = float_type
        self.hidden_dim = hidden_dim
        self.vector_size = vector_size
        self.attention_length = attention_length
        self.pointer_gen = pointer_gen
        self.use_coverage = use_coverage
        self.initial_state_attention = initial_state_attention

    def build(self, unused_input_shapes):

        self.lstm_layer = tf.keras.layers.LSTMCell(self.vector_size)

        self.encoder_layer = tf.keras.layers.Conv2D(filters=self.vector_size, kernel_size=(1, 1), padding="SAME")
        # shape (batch_size,attn_length,1,attention_vec_size)
        self.linear = LinearLayer(self.vector_size)
        self.linear2 = LinearLayer(1)
        self.attention_layer = AttentionLayer(self.vector_size, self.attention_length, True)

    def __call__(self,
                 decoder_inputs,
                 dec_initial_state,
                 encoder_states,
                 enc_padding_mask,
                 prev_coverage=None,
                 **kwargs):

        inputs = (decoder_inputs,
                  dec_initial_state,
                  encoder_states,
                  enc_padding_mask,
                  prev_coverage)
        return super(OneStepDecodeLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        # unpacked_inputs = tf_utils.unpack_inputs(inputs)
        decoder_inputs = inputs[0]
        initial_state = inputs[1]
        encoder_states = inputs[2]
        enc_padding_mask = inputs[3]
        prev_coverage = inputs[4]

        outputs = []
        attn_dists = []
        p_gens = []

        encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

        encoder_features = self.encoder_layer(encoder_states)  # shape (batch_size,attn_length,1,attention_vec_size)
        state = initial_state  # [initial_state,initial_state]
        # state=[initial_state]*2
        batch_size = tf_utils.get_shape_list(encoder_states)[0]

        coverage = prev_coverage  # initialize coverage to None or whatever was passed in

        context_vector = tf.zeros([batch_size, self.vector_size])

        # Merge input and previous attentions into one vector x of the same size as inp
        input_size = decoder_inputs.get_shape().with_rank(2)[1]

        if input_size is None:
            raise ValueError("Could not infer input size from input: %s" % decoder_inputs.name)

        x = self.linear([[decoder_inputs], [context_vector]])

        # Run the decoder RNN cell. cell_output = decoder state
        # print(i, x, state)
        cell_output, state = self.lstm_layer(x, state)

        # Run the attention mechanism.
        context_vector, attn_dist, coverage = self.attention_layer(encoder_features=encoder_features,
                                                                       decoder_state=state,
                                                                       coverage=coverage,
                                                                       input_mask=enc_padding_mask)
        attn_dists.append(attn_dist)

        # Calculate p_gen
        if self.pointer_gen:
            p_gen = self.linear2([[context_vector], [state[0]], [state[1]], [x]])
            # Tensor shape (batch_size, 1)
            p_gen = tf.sigmoid(p_gen)
            p_gens.append(p_gen)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
        output = self.linear([[cell_output], [context_vector]])

        outputs.append(output)

        return outputs, state, attn_dists, p_gens, coverage

class PGNetBeamSerachLayer(tf.keras.layers.Layer):
    def __init__(self,
                 config, training=False,
                 float_type=tf.float32,
                 **kwargs):
        super(PGNetBeamSerachLayer, self).__init__(**kwargs)

        self.config = (
            PGNetConfig.from_dict(config)
            if isinstance(config, dict) else copy.deepcopy(config))

        self.float_type = float_type
        self.training = training

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.embedding_lookup = EmbeddingLookup(self.config.vocab_size,
                                                self.config.hidden_size,
                                                dtype=tf.float32,
                                                )
        self.encoder = Encoder(self.config.hidden_size, self.config.max_seq_length, dynamic=True)
        self.decoder = OneStepDecodeLayer(self.config.hidden_size, self.config.hidden_size,
                                        self.config.max_seq_length, get_initializer())
        self.output_projector = OutputProjectionLayer(self.config.hidden_size, self.config.vocab_size)
        self.final_distribution = FinalDistributionLayer(self.config.hidden_size, self.config.vocab_size,
                                                         self.config.max_oov_size)

        super(PGNetBeamSerachLayer, self).build(unused_input_shapes)

    def __call__(self,
                 input_ids,
                 input_mask=None,
                 answer_ids=None,
                 answer_mask=None,
                 **kwargs):
        if type(input_ids) is tuple:
            inputs = input_ids
        else:
            inputs = (input_ids, input_mask, answer_ids, answer_mask)
        return super(PGNetBeamSerachLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs):

        input_ids, input_mask, answer_ids, answer_mask = inputs
        emb_enc_inputs = self.embedding_lookup(input_ids)
        # tensor with shape (batch_size, max_seq_length, emb_size)

        emb_dec_inputs = [self.embedding_lookup(x) for x in tf.unstack(answer_ids,
                                                                       axis=1)]
        # list length max_dec_steps containing shape (batch_size, emb_size)

        enc_outputs, enc_state = self.encoder(emb_enc_inputs, input_mask)

        _enc_states = enc_outputs

        _dec_in_state = enc_state

        atten_len=tf_utils.get_shape_list(input_ids)[1]
        batch_size = tf_utils.get_shape_list(input_ids)[0]

        UNKNOWN_TOKEN = 0
        START_TOKEN = 104
        STOP_TOKEN = 105

        prev_coverage =   self.prev_coverage #if self.config.mode == "decode" and self.config.use_coverage  else None
        hyps = [Hypothesis(tokens=[START_TOKEN],  #answer_ids[0] is the [START]
                           log_probs=[0.0],
                           state=_dec_in_state,
                           attn_dists=[],
                           p_gens=[],
                           coverage=tf.zeros(atten_len)  # zero vector of length attention_length
                           ) for _ in tf.range(batch_size)]

        results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)



        steps = 0
        while steps < self.config.max_dec_steps and len(results) < self.config.beam_size:
            latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
            latest_tokens = [t if t in tf.range(self.config.vocab_size) else UNKNOWN_TOKEN for t in
                             latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
            states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
            prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)

            # Run one step of the decoder to get the new info
            decoder_outputs, new_states, attn_dists, p_gens, new_coverage = self.decoder(
                latest_tokens,
                _dec_in_state,
                states,
                input_mask,
                prev_coverage=prev_coverage)

            vocab_dists = self.output_projector(decoder_outputs)

            if self.config.use_pointer_gen:
                final_dists = self.final_distribution(vocab_dists, attn_dists, p_gens, input_ids)
            else:  # final distribution is just vocabulary distribution
                final_dists = vocab_dists


            assert len(
                final_dists) == 1  # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            final_dists = final_dists[0]
            topk_probs, topk_ids = tf.nn.top_k(final_dists,
                                                     self.config.batch_size * 2)
            # take the k largest probs. note batch_size=beam_size in decode mode

            topk_log_probs = tf.log(topk_probs)

            # Extend each hypothesis and collect them all in all_hyps
            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len( hyps)
            # On the first step, we only had one original hypothesis (the initial hypothesis).
            # On subsequent steps, all original hypotheses are distinct.

            for i in tf.range(num_orig_hyps):
                h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                                 new_coverage[ i]
                # take the ith hypothesis and new decoder state info

                for j in tf.range(self.config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    # Extend the ith hypothesis with the jth option
                    new_hyp = h.extend(token=topk_ids[i, j],
                                       log_prob=topk_log_probs[i, j],
                                       state=new_state,
                                       attn_dist=attn_dist,
                                       p_gen=p_gen,
                                       coverage=new_coverage_i)
                    all_hyps.append(new_hyp)

            # Filter and collect any hypotheses that have produced the end token.
            hyps = []  # will contain hypotheses for the next step
            for h in self.sort_hyps(all_hyps):  # in order of most likely h
                if h.latest_token == STOP_TOKEN:  # if stop token is reached...
                    # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                    if steps >= self.config.min_dec_steps:
                        results.append(h)
                else:  # hasn't reached stop token, so continue to extend this hypothesis
                    hyps.append(h)
                if len(hyps) == self.config.beam_size or len(results) == self.config.beam_size:
                    # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                    break

            steps += 1

            # At this point, either we've got beam_size results, or we've reached maximum decoder steps

        if len(
                results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
            results = hyps

            # Sort hypotheses by average log probability
        hyps_sorted = self.sort_hyps(results)

        # Return the hypothesis with highest average log prob
        return hyps_sorted[0]

    def sort_hyps(self,hyps):
        """Return a list of Hypothesis objects, sorted by descending average log probability"""
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)

class PGNetSummaryModel(tf.keras.layers.Layer):
    def __init__(self,
                 config,training=False,
                 float_type=tf.float32,
                 **kwargs):
        super(PGNetSummaryModel, self).__init__(**kwargs)

        self.config = (
            PGNetConfig.from_dict(config)
            if isinstance(config, dict) else copy.deepcopy(config))

        self.float_type = float_type
        self.training = training

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.embedding_lookup = EmbeddingLookup(self.config.vocab_size,
                                                self.config.hidden_size,
                                                dtype=tf.float32,
                                                )
        self.encoder = Encoder(self.config.hidden_size, self.config.max_seq_length, dynamic=True)
        self.decoder = AttentionDecoder(self.config.hidden_size, self.config.hidden_size,
                                        self.config.max_seq_length, get_initializer())
        self.output_projector = OutputProjectionLayer(self.config.hidden_size, self.config.vocab_size)
        self.final_distribution = FinalDistributionLayer(self.config.hidden_size, self.config.vocab_size,
                                                         self.config.max_oov_size)

        super(PGNetSummaryModel, self).build(unused_input_shapes)

    def __call__(self,
                 input_ids,
                 input_mask=None,
                 answer_ids=None,
                 answer_mask=None,
                 **kwargs):
        if type(input_ids) is tuple:
            inputs=input_ids
        else:
            inputs = (input_ids, input_mask, answer_ids, answer_mask)
        return super(PGNetSummaryModel, self).__call__(inputs, **kwargs)

    def call(self, inputs, mode="decode"):

        input_ids ,input_mask ,answer_ids ,answer_mask = inputs
        emb_enc_inputs = self.embedding_lookup(input_ids)
        # tensor with shape (batch_size, max_seq_length, emb_size)

        emb_dec_inputs = [self.embedding_lookup(x) for x in tf.unstack(answer_ids,
                                                                       axis=1)]
        # list length max_dec_steps containing shape (batch_size, emb_size)

        enc_outputs, enc_state = self.encoder(emb_enc_inputs, input_mask)

        _enc_states = enc_outputs

        _dec_in_state = enc_state

        # if mode == "encoder":
        #     return (_enc_states, _dec_in_state)

        prev_coverage = None  # self.prev_coverage #if self.config.mode == "decode" and self.config.use_coverage  else None

        decoder_outputs, _dec_out_state, attn_dists, p_gens, coverage = self.decoder(
            emb_dec_inputs,
            _dec_in_state,
            _enc_states,
            input_mask,
            prev_coverage=prev_coverage)
        # if mode == "decoder":
        #     return (decoder_outputs, _dec_out_state, attn_dists, p_gens, coverage)

        vocab_dists = self.output_projector(decoder_outputs)

        if self.config.use_pointer_gen:
            final_dists = self.final_distribution(vocab_dists, attn_dists, p_gens, input_ids)
        else:  # final distribution is just vocabulary distribution
            final_dists = vocab_dists

        return final_dists, attn_dists

    def get_config(self):
        config = {"config": self.config.to_dict()}
        base_config = super(PGNetSummaryModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EmbeddingLookup(tf.keras.layers.Layer):
  """Looks up words embeddings for id tensor."""

  def __init__(self,
               vocab_size,
               embedding_size=768,
               initializer_range=0.02,
               **kwargs):
    super(EmbeddingLookup, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.initializer_range = initializer_range

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.vocab_size, self.embedding_size],
        initializer=get_initializer(self.initializer_range),
        dtype=self.dtype)
    super(EmbeddingLookup, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""

    input_shape = tf_utils.get_shape_list(inputs)

    flat_input = tf.reshape(inputs, [-1])
    output = tf.gather(self.embeddings, flat_input)
    output = tf.reshape(output, input_shape + [self.embedding_size])


    return output

class EmbeddingPostprocessor(tf.keras.layers.Layer):
  """Performs various post-processing on a word embedding tensor."""

  def __init__(self,
               use_type_embeddings=False,
               token_type_vocab_size=None,
               use_position_embeddings=True,
               max_position_embeddings=512,
               dropout_prob=0.0,
               initializer_range=0.02,
               initializer=None,
               **kwargs):
    super(EmbeddingPostprocessor, self).__init__(**kwargs)
    self.use_type_embeddings = use_type_embeddings
    self.token_type_vocab_size = token_type_vocab_size
    self.use_position_embeddings = use_position_embeddings
    self.max_position_embeddings = max_position_embeddings
    self.dropout_prob = dropout_prob
    self.initializer_range = initializer_range

    if not initializer:
      self.initializer = get_initializer(self.initializer_range)
    else:
      self.initializer = initializer

    if self.use_type_embeddings and not self.token_type_vocab_size:
      raise ValueError("If `use_type_embeddings` is True, then "
                       "`token_type_vocab_size` must be specified.")

  def build(self, input_shapes):
    """Implements build() for the layer."""
    (word_embeddings_shape, _) = input_shapes
    width = word_embeddings_shape.as_list()[-1]
    self.type_embeddings = None
    if self.use_type_embeddings:
      self.type_embeddings = self.add_weight(
          "type_embeddings",
          shape=[self.token_type_vocab_size, width],
          initializer=get_initializer(self.initializer_range),
          dtype=self.dtype)

    self.position_embeddings = None
    if self.use_position_embeddings:
      self.position_embeddings = self.add_weight(
          "position_embeddings",
          shape=[self.max_position_embeddings, width],
          initializer=get_initializer(self.initializer_range),
          dtype=self.dtype)

    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_prob,
                                                  dtype=tf.float32)
    super(EmbeddingPostprocessor, self).build(input_shapes)

  def __call__(self, word_embeddings, token_type_ids=None, **kwargs):
    inputs = tf_utils.pack_inputs([word_embeddings, token_type_ids])
    return super(EmbeddingPostprocessor, self).__call__(inputs, **kwargs)

  def call(self, inputs):
        """Implements call() for the layer."""
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        word_embeddings = unpacked_inputs[0]
        token_type_ids = unpacked_inputs[1]
        input_shape = tf_utils.get_shape_list(word_embeddings, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        output = word_embeddings
        if self.use_type_embeddings:
          flat_token_type_ids = tf.reshape(token_type_ids, [-1])
          token_type_embeddings = tf.gather(self.type_embeddings,
                                            flat_token_type_ids)
          token_type_embeddings = tf.reshape(token_type_embeddings,
                                             [batch_size, seq_length, width])
          output += token_type_embeddings

        if self.use_position_embeddings:
          position_embeddings = tf.expand_dims(
              tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
              axis=0)

          output += position_embeddings

        output = self.output_layer_norm(output)
        output = self.output_dropout(output)

        return output

class PositionalEmbeddingLookup(tf.keras.layers.Layer):
  """Looks up words embeddings for id tensor."""

  def __init__(self,
               seq_length,
               embedding_size=768,
               initializer_range=0.02,
               **kwargs):
    super(PositionalEmbeddingLookup, self).__init__(**kwargs)

    self.seq_length = seq_length
    self.embedding_size = embedding_size
    self.initializer_range = initializer_range

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.seq_length, self.embedding_size],
        initializer=get_initializer(self.initializer_range),
        dtype=self.dtype)
    super(PositionalEmbeddingLookup, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""

    input_shape = tf_utils.get_shape_list(inputs)
    batch_size = input_shape[0]

    pos_ind = tf.tile(tf.expand_dims(tf.range(self.seq_length), 0), [batch_size, 1])

    output = tf.gather(self.embeddings, pos_ind)
    output = tf.reshape(output, input_shape + [self.embedding_size])


    return output


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim, max_seq_length, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

    def build(self, unused_input_shapes):
        lstm_layer_fw = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        lstm_layer_bw = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True, go_backwards=True,
                                             return_state=True)
        self.bidirection = tf.keras.layers.Bidirectional(lstm_layer_fw, backward_layer=lstm_layer_bw,
                                                         merge_mode="concat")
        self.dense = tf.keras.layers.Dense(self.hidden_dim)

        self.state_reducer = ReduceStateLayer(self.hidden_dim)

        super(Encoder, self).build(unused_input_shapes)

    def __call__(self,
                 input_ids,
                 input_mask,
                 **kwargs):
        inputs = (input_ids, input_mask)
        return super(Encoder, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        # unpacked_inputs = tf_utils.unpack_inputs(inputs)
        input_ids = inputs[0]
        masks = inputs[1]
        masks = tf.expand_dims(masks, axis=2)

        outputs = self.bidirection(input_ids * tf.cast(masks,dtype=tf.float32))

        encoder_outputs = self.dense(outputs[0])

        fw_state_h, fw_state_c = outputs[1], outputs[2]
        bw_state_h, bw_state_c = outputs[3], outputs[4]

        state = self.state_reducer(fw_state_h, fw_state_c, bw_state_h, bw_state_c)

        return encoder_outputs, state

    def compute_output_shape(self, inputShape):
        # calculate shapes from input shape
        return [tf.TensorShape((None, self.max_seq_length,  self.hidden_dim)),
                [tf.TensorShape((None, self.hidden_dim)), tf.TensorShape((None, self.hidden_dim))]]


class ReduceStateLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim, **kwargs):
        super(ReduceStateLayer, self).__init__(**kwargs)
        self.hidden_dim=hidden_dim

    def build(self, unused_input_shapes):
        hidden_dim = self.hidden_dim
        self.w_reduce_c = self.add_weight('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal())
        self.w_reduce_h = self.add_weight('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal())
        self.bias_reduce_c = self.add_weight('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                        initializer=tf.keras.initializers.TruncatedNormal())
        self.bias_reduce_h = self.add_weight('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                        initializer=tf.keras.initializers.TruncatedNormal())
        super(ReduceStateLayer, self).build(unused_input_shapes)

    def __call__(self,
                 fw_state_h,fw_state_c, bw_state_h,bw_state_c,
                 **kwargs):
        inputs =  (fw_state_h,fw_state_c, bw_state_h,bw_state_c)

        return super(ReduceStateLayer, self).__call__(inputs, **kwargs)
    def call(self, inputs):

        fw_state_h = inputs[0]
        fw_state_c = inputs[1]
        bw_state_h = inputs[2]
        bw_state_c = inputs[3]

        # Apply linear layer
        old_c = tf.concat(axis=1, values=[fw_state_c, bw_state_c])  # Concatenation of fw and bw cell
        old_h = tf.concat(axis=1, values=[fw_state_h, bw_state_h])  # Concatenation of fw and bw state
        new_c = tf.nn.relu(tf.matmul( old_c, self.w_reduce_c) + self.bias_reduce_c)  # Get new cell from old cell
        new_h = tf.nn.relu(tf.matmul( old_h, self.w_reduce_h) + self.bias_reduce_h)  # Get new state from old state
        return [new_c, new_h]  # Return new cell and state

    def compute_output_shape(self, inputShape):
        # calculate shapes from input shape
        return [[None, self.config.hidden_size],
                [None, self.config.hidden_size]]



class OutputProjectionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 vocab_size,
                 **kwargs):
        super(OutputProjectionLayer, self).__init__(**kwargs)
        self.hidden_dim=hidden_dim
        self.vocab_size =vocab_size


    def build(self, unused_input_shapes):
        self.w = self.add_weight('w', [self.hidden_dim, self.vocab_size], dtype=tf.float32, initializer=tf.keras.initializers.TruncatedNormal())
        self.w_t = tf.transpose(self.w)
        self.v = self.add_weight('v', [self.vocab_size], dtype=tf.float32, initializer=tf.keras.initializers.TruncatedNormal())
        super(OutputProjectionLayer, self).build(unused_input_shapes)


    def call(self, inputs):
        decoder_outputs = inputs

        vocab_scores = []  # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i, output in enumerate(decoder_outputs):
            vocab_scores.append(tf.matmul(output, self.w)+ self.v)  # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for s in
                       vocab_scores]
        # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

        return vocab_dists


class FinalDistributionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 vocab_size,
                 max_oov_size,
                 **kwargs):
        super(FinalDistributionLayer, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_oov_size = max_oov_size

    # def build(self, unused_input_shapes):

    def __call__(self,
                 vocab_dists, attn_dists, p_gens, input_ids,
                 **kwargs):
        inputs = (vocab_dists, attn_dists, p_gens, input_ids)

        return super(FinalDistributionLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        vocab_dists = inputs[0]
        attn_dists = inputs[1]
        p_gens = inputs[2]
        input_ids = inputs[3]
        max_oov_size = self.max_oov_size

        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

        batch_size = tf_utils.get_shape_list(vocab_dists[0])[0]

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        extended_vsize = self.vocab_size + max_oov_size  # the maximum (over the batch) size of the extended vocabulary
        extra_zeros = tf.zeros((batch_size, max_oov_size))
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
                                vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        attn_len = tf_utils.get_shape_list(input_ids)[1]  # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
        indices = tf.stack((batch_nums, input_ids), axis=2)  # shape (batch_size, enc_t, 2)
        shape = [batch_size, extended_vsize]

        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
                                attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                       zip(vocab_dists_extended, attn_dists_projected)]

        return final_dists

class AttentionDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 vector_size,
                 attention_length,
                 initial_state_attention=False,
                 pointer_gen=True,
                 use_coverage=False,
                 initializer=None,
                 float_type=tf.float32,
                 **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
        self.initializer = initializer
        self.float_type = float_type
        self.hidden_dim = hidden_dim
        self.vector_size = vector_size
        self.attention_length = attention_length
        self.pointer_gen = pointer_gen
        self.use_coverage = use_coverage
        self.initial_state_attention = initial_state_attention

    def build(self, unused_input_shapes):

        self.lstm_layer = tf.keras.layers.LSTMCell(self.vector_size)

        self.encoder_layer = tf.keras.layers.Conv2D(filters=self.vector_size, kernel_size=(1, 1), padding="SAME")
        # shape (batch_size,attn_length,1,attention_vec_size)
        self.linear = LinearLayer(self.vector_size)
        self.linear2 = LinearLayer(1)
        self.attention_layer = AttentionLayer(self.vector_size, self.attention_length, True)

    def __call__(self,
                 decoder_inputs,
                 dec_initial_state,
                 encoder_states,
                 enc_padding_mask,
                 prev_coverage=None,
                 **kwargs):

        inputs = (decoder_inputs,
                  dec_initial_state,
                  encoder_states,
                  enc_padding_mask,
                  prev_coverage)
        return super(AttentionDecoder, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        # unpacked_inputs = tf_utils.unpack_inputs(inputs)

        decoder_inputs = inputs[0]
        initial_state = inputs[1]
        encoder_states = inputs[2]
        enc_padding_mask = inputs[3]
        prev_coverage = inputs[4]

        outputs = []
        attn_dists = []
        p_gens = []

        encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

        encoder_features = self.encoder_layer(encoder_states)  # shape (batch_size,attn_length,1,attention_vec_size)
        state = initial_state  # [initial_state,initial_state]
        # state=[initial_state]*2
        batch_size = tf_utils.get_shape_list(encoder_states)[0]

        coverage = prev_coverage  # initialize coverage to None or whatever was passed in

        context_vector = tf.zeros([batch_size, self.vector_size])
        context_vector.set_shape([None, self.vector_size])  # Ensure the second shape of attention vectors is set.
        if self.initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step so that we can pass it through a linear layer
            # with this step's input to get a modified version of the input
            context_vector, _, coverage = self.attention_layer(encoder_features=encoder_features,
                                                               decoder_state=state,
                                                               coverage=coverage,
                                                               input_mask=enc_padding_mask)
            # in decode mode, this is what updates the coverage vector

        for i, inp in enumerate(decoder_inputs):

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]

            if input_size is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)


            x = self.linear([[inp], [context_vector]])

            # Run the decoder RNN cell. cell_output = decoder state
            # print(i, x, state)
            cell_output, state = self.lstm_layer(x, state)

            # Run the attention mechanism.
            if i == 0 and self.initial_state_attention:  # always true in decode mode
                context_vector, attn_dist, _ = self.attention_layer(encoder_features=encoder_features,
                                                                    decoder_state=state,
                                                                    coverage=coverage,
                                                                    input_mask=enc_padding_mask)  # don't allow coverage to update
            else:
                context_vector, attn_dist, coverage = self.attention_layer(encoder_features=encoder_features,
                                                                           decoder_state=state,
                                                                           coverage=coverage,
                                                                           input_mask=enc_padding_mask)
            attn_dists.append(attn_dist)

            # Calculate p_gen
            if self.pointer_gen:
                p_gen = self.linear2([[context_vector], [state[0]], [state[1]], [x]])
                # Tensor shape (batch_size, 1)
                p_gen = tf.sigmoid(p_gen)
                p_gens.append(p_gen)

                # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
                # This is V[s_t, h*_t] + b in the paper
                output = self.linear([[cell_output], [context_vector]])
            outputs.append(output)

        # If using coverage, reshape it
        if coverage is not None:
            coverage = tf.reshape(coverage, [batch_size, -1])

        return outputs, state, attn_dists, p_gens, coverage


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, vector_size,
                 use_coverage,
                 initializer=None,
                 float_type=tf.float32,
                 **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.initializer = initializer
        self.float_type = float_type
        self.vector_size = vector_size
        self.use_coverage = use_coverage

        # self.w_h = self.add_weight(shape=[1, 1, self.attention_length, self.vector_size], name="W_h")
        self.v = self.add_weight(shape=[self.vector_size], name="v")
        # self.w_c = self.add_weight(shape=[1, 1, 1, self.vector_size], name="W_c")

    def build(self, unused_input_shapes):

        self.linear_layer = LinearLayer(self.vector_size)
        # shape (batch_size, attention_vec_size)

        self.coverage_layer = tf.keras.layers.Conv2D(self.vector_size, (1, 1), padding="SAME")
        # c has shape (batch_size, attn_length, 1, attention_vec_size)

        super(AttentionLayer, self).build(unused_input_shapes)

    def __call__(self,
                 decoder_state,
                 encoder_features,
                 input_mask,
                 coverage=None,
                 **kwargs):
        inputs = (encoder_features, decoder_state, input_mask, coverage)
        return super(AttentionLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs):

        encoder_features = inputs[0]
        batch_size = tf_utils.get_shape_list(encoder_features)[0]

        decoder_states = inputs[1]
        input_mask = inputs[2]
        coverage = inputs[3]

        decoder_features = self.linear_layer([decoder_states])
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

        # reshape to (batch_size, 1, 1, attention_vec_size)

        def masked_attention(e):
            """Take softmax of e then apply enc_padding_mask and re-normalize"""
            attn_dist = tf.nn.softmax(e)  # take softmax. shape (batch_size, attn_length)
            attn_dist *= tf.dtypes.cast(input_mask,tf.float32)  # apply mask
            masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
            return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

        if self.use_coverage and coverage is not None:  # non-first step of coverage
            # Multiply coverage vector by w_c to get coverage_features.
            coverage_features = self.coverage_layer(
                coverage)  # c has shape (batch_size, attn_length, 1, attention_vec_size)

            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
            e = tf.reduce_sum(self.v * tf.tanh(encoder_features + decoder_features + coverage_features),
                              [2, 3])  # shape (batch_size,attn_length)

            # Calculate attention distribution
            attn_dist = masked_attention(e)

            # Update coverage vector
            coverage += tf.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            e = tf.reduce_sum(self.v * tf.tanh(encoder_features + decoder_features), [2, 3])  # calculate e

            # Calculate attention distribution
            attn_dist = masked_attention(e)

            if self.use_coverage:  # first step of training
                coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage

        # Calculate the context vector from attn_dist and encoder_states
        context_vector = tf.reduce_sum(tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_features,
                                       [1, 2])  # shape (batch_size, attn_size).

        context_vector = tf.reshape(context_vector, [-1, self.vector_size])

        return context_vector, attn_dist, coverage

    def compute_output_shape(self, inputShape):

        # calculate shapes from input shape
        return [[None, self.vector_size],
                [None, self.max_seq_length],
                [None, self.max_seq_length, 1, 1]
                ]


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                 activation=None,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        """Implements build() for the layer."""

        total_arg_size = 0
        shapes = input_shape
        if type(shapes) is not list:
            shapes = [shapes]

        shapes = list(itertools.chain(*shapes))

        for shape in shapes:

            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]


        self.kernel = self.add_weight(
            "kernel",
            shape=[total_arg_size, self.output_size],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)

        super(LinearLayer, self).build(input_shape)

    def call(self, inputs):

        if type(inputs) is not list:
            inputs = [inputs]

        inputs = list(itertools.chain(*inputs))

        if len(inputs) == 1:

            res = tf.matmul(inputs[0], self.kernel)
        else:
            res = tf.matmul(tf.concat(axis=1, values=inputs), self.kernel)

        if not self.use_bias:
            return res

        return res + self.bias


class SimpleAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, vector_size,
                 initializer=None,
                 float_type=tf.float32,
                 **kwargs):
        super(SimpleAttentionLayer, self).__init__(**kwargs)
        self.initializer = initializer
        self.float_type = float_type
        self.vector_size = vector_size
        self.v = self.add_weight(shape=[self.vector_size], name="v")

    def build(self, unused_input_shapes):

        self.linear_layer = LinearLayer(self.vector_size)
        # shape (batch_size, attention_vec_size)

        super(SimpleAttentionLayer, self).build(unused_input_shapes)

    def __call__(self,
                 decoder_state,
                 encoder_features=None,
                 input_mask=None,
                 **kwargs):
        if type(decoder_state) is tuple:
            inputs = decoder_state
        else:
            inputs = (encoder_features, decoder_state, input_mask)
        return super(SimpleAttentionLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs):

        encoder_features = inputs[0]
        batch_size = tf_utils.get_shape_list(encoder_features)[0]

        decoder_states = inputs[1]
        input_mask = inputs[2]

        decoder_features = self.linear_layer([decoder_states])
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

        # reshape to (batch_size, 1, 1, attention_vec_size)

        def masked_attention(e):
            """Take softmax of e then apply enc_padding_mask and re-normalize"""
            attn_dist = tf.nn.softmax(e)  # take softmax. shape (batch_size, attn_length)
            attn_dist *= tf.dtypes.cast(input_mask,tf.float32)  # apply mask
            masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
            return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize


            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
        e = tf.reduce_sum(self.v * tf.tanh(encoder_features + decoder_features),
                          [2, 3])  # shape (batch_size,attn_length)

        # Calculate attention distribution
        attn_dist = masked_attention(e)


        # Calculate the context vector from attn_dist and encoder_states
        context_vector = tf.reduce_sum(tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_features,
                                       [1, 2])  # shape (batch_size, attn_size).

        context_vector = tf.reshape(context_vector, [-1, self.vector_size])

        return context_vector



class SimpleLSTMSeq2Seq(tf.keras.layers.Layer):
    def __init__(self,
                 config, training=True, **kwargs):
        super(SimpleLSTMSeq2Seq, self).__init__(**kwargs)
        self.config = config
        self.training = training

    def build(self, unused_input_shapes):

        self.embedding_lookup = EmbeddingLookup(self.config.vocab_size,
                                                self.config.hidden_size,
                                                dtype=tf.float32,
                                                name = "embedding"
                                                )
        self.embedding_postprocessor = EmbeddingPostprocessor(
            use_type_embeddings=True,
            token_type_vocab_size=self.config.type_vocab_size, #4, #we have four types of tokens (question,story,previous question, previous answer)
            use_position_embeddings=False,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_prob=self.config.hidden_dropout_prob,
            initializer_range=self.config.initializer_range,
            dtype=tf.float32,
            name="embedding_postprocessor")

        self.lstm = tf.keras.layers.LSTM(self.config.hidden_size,return_sequences=True,return_state = True,name='encoder')
        self.encoder = tf.keras.layers.Conv2D(filters=self.config.hidden_size, kernel_size=(1, 1), padding="SAME")
        self.attentioner = SimpleAttentionLayer(vector_size=self.config.hidden_size)
        self.linear = LinearLayer(self.config.hidden_size)
        self.decoder_cell = tf.keras.layers.LSTMCell(self.config.hidden_size,name="decoder")
        self.output_projector = OutputProjectionLayer(self.config.hidden_size, self.config.vocab_size,name="projector")

        super(SimpleLSTMSeq2Seq, self).build(unused_input_shapes)

    def __call__(self,
                 input_ids,
                 input_mask=None,
                 input_type_ids= None,
                 decode_ids=None,
                 **kwargs):

        if type(input_ids) is tuple:
            inputs = input_ids
        else:
            inputs = (input_ids, input_mask, input_type_ids, decode_ids)
        return super(SimpleLSTMSeq2Seq, self).__call__(inputs, **kwargs)

    def call(self, inputs ):
        # unpacked_inputs = tf_utils.unpack_inputs(inputs)
        (input_ids, input_mask,input_type_ids, decode_ids) =inputs

        #input_mask = tf.expand_dims(input_mask, axis=2)

        emb_enc_inputs = self.embedding_lookup(input_ids)

        embedding_tensor = self.embedding_postprocessor(
            word_embeddings=emb_enc_inputs, token_type_ids=input_type_ids)

        emb_dec_inputs = [self.embedding_lookup(x) for x in tf.unstack(decode_ids,
                                                                       axis=1)]
        #self.encoder.reset_states()
        enc_outputs,state_h,state_c  = self.lstm(embedding_tensor,mask=tf.cast(input_mask,dtype=tf.bool))

        states=[state_h,state_c]

        enc_outputs = tf.expand_dims(enc_outputs,2)

        enc_features= self.encoder(enc_outputs)

        outputs=[]

        for i, inp in enumerate(emb_dec_inputs): #this is simply a plain LSTM but we unrolled it.

            cell_output, states = self.decoder_cell(inp, states)

            context = self.attentioner(states, enc_features, input_mask)
            output= self.linear([[cell_output],[context]])

            outputs.append(output)

        vocab_dists = self.output_projector(outputs)

        return vocab_dists

    def get_config(self):
        config = {"config": self.config.to_dict()}
        base_config = super(SimpleLSTMSeq2Seq, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def compute_output_shape(self, inputShape):
    #     # calculate shapes from input shape
    #     return [tf.TensorShape((None, self.config.max_answer_length , self.config.vocab_size))]


class SimpleTransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, config, float_type=tf.float32, **kwargs):
        super(SimpleTransformerDecoder, self).__init__(**kwargs)
        self.config =  config
        self.float_type = float_type

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.embedding_lookup = EmbeddingLookup(
            vocab_size=self.config.vocab_size,
            embedding_size=self.config.hidden_size,
            initializer_range=self.config.initializer_range,
            dtype=tf.float32,
            name="word_embeddings")

        self.decoder =  DecodeTransformer(
            num_hidden_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_activation=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            backward_compatible=self.config.backward_compatible,
            float_type=self.float_type,
            name="decoder")

        self.dense = tf.keras.layers.Dense(self.config.vocab_size)

        self.position_embedding = PositionalEmbeddingLookup(self.config.max_answer_length)

        super(SimpleTransformerDecoder, self).build(unused_input_shapes)

    def __call__(self,
                 sequence_output,
                 input_mask=None,
                 target_ids = None,
                 target_mask = None,
                 **kwargs):
        if type(sequence_output) is tuple:
            inputs = sequence_output
        else:
            inputs = (sequence_output, input_mask, target_ids,target_mask)

        #inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids,target_ids,target_mask])
        return super(SimpleTransformerDecoder, self).__call__(inputs, **kwargs)

    def call(self, inputs ):

        (sequence_output, input_mask, target_ids, target_mask) =inputs


        target_attention_mask =None
        if target_mask is not None:
            target_attention_mask = bert_modeling.create_attention_mask_from_input_mask(
                target_ids, target_mask)
            target_attention_mask *= construct_autoregressive_mask(target_ids)


        if input_mask is not None:
            decode_attention_mask = bert_modeling.create_attention_mask_from_input_mask(
                target_ids, input_mask)

        # Target embedding + positional encoding
        dec_inp_embed =  self.embedding_lookup(target_ids )
        dec_inp_embed = dec_inp_embed + self.position_embedding(target_ids)
        #
        dec_out = self.decoder( sequence_output, decode_attention_mask,dec_inp_embed, target_attention_mask)
        #
        # # Make the prediction out of the decoder output.
        logits = self.dense(dec_out)  # [batch, target_vocab]

        return  logits


class SimpleTransformer(tf.keras.layers.Layer):
    def __init__(self, config, float_type=tf.float32, **kwargs):
        super(SimpleTransformer, self).__init__(**kwargs)
        self.config =  config
        self.float_type = float_type

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.embedding_lookup = EmbeddingLookup(
            vocab_size=self.config.vocab_size,
            embedding_size=self.config.hidden_size,
            initializer_range=self.config.initializer_range,
            dtype=tf.float32,
            name="word_embeddings")
        self.embedding_postprocessor = EmbeddingPostprocessor(
            use_type_embeddings=True,
            token_type_vocab_size=  self.config.type_vocab_size,
            use_position_embeddings=True,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_prob=self.config.hidden_dropout_prob,
            initializer_range=self.config.initializer_range,
            dtype=tf.float32,
            name="embedding_postprocessor")
        self.encoder = bert_modeling.Transformer(
            num_hidden_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_activation=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            backward_compatible=self.config.backward_compatible,
            float_type=self.float_type,
            name="encoder")

        self.decoder =  DecodeTransformer(
            num_hidden_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_activation=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            backward_compatible=self.config.backward_compatible,
            float_type=self.float_type,
            name="decoder")

        self.pooler_transform = tf.keras.layers.Dense(
            units=self.config.hidden_size,
            activation="tanh",
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="pooler_transform")

        self.dense = tf.keras.layers.Dense(self.config.vocab_size)

        self.position_embedding = PositionalEmbeddingLookup(self.config.max_answer_length)

        super(SimpleTransformer, self).build(unused_input_shapes)

    def __call__(self,
                 input_word_ids,
                 input_mask=None,
                 input_type_ids=None,
                 target_ids = None,
                 target_mask = None,
                 **kwargs):
        if type(input_word_ids) is tuple:
            inputs = input_word_ids
        else:
            inputs = (input_word_ids, input_mask, input_type_ids, target_ids,target_mask)

        #inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids,target_ids,target_mask])
        return super(SimpleTransformer, self).__call__(inputs, **kwargs)

    def call(self, inputs ):
        """Implements call() for the layer.

        Args:
          inputs: packed input tensors.
          mode: string, `coqa` or `encoder`.
        Returns:

        """
        #unpacked_inputs = tf_utils.unpack_inputs(inputs)
        # input_word_ids = unpacked_inputs[0]
        # input_mask = unpacked_inputs[1]
        # input_type_ids = unpacked_inputs[2]
        # target_ids = unpacked_inputs[3]
        # target_mask =  unpacked_inputs[4]

        (input_word_ids, input_mask, input_type_ids, target_ids, target_mask) =inputs
        word_embeddings = self.embedding_lookup(input_word_ids)
        embedding_tensor = self.embedding_postprocessor(
            word_embeddings=word_embeddings, token_type_ids=input_type_ids)
        if self.float_type == tf.float16:
            embedding_tensor = tf.cast(embedding_tensor, tf.float16)
        attention_mask = None
        if input_mask is not None:
            attention_mask = bert_modeling.create_attention_mask_from_input_mask(
                input_word_ids, input_mask)

        target_attention_mask =None
        if target_mask is not None:
            target_attention_mask = bert_modeling.create_attention_mask_from_input_mask(
                target_ids, target_mask)
            target_attention_mask *= construct_autoregressive_mask(target_ids)

        sequence_output = self.encoder(embedding_tensor, attention_mask)

        if input_mask is not None:
            decode_attention_mask = bert_modeling.create_attention_mask_from_input_mask(
                target_ids, input_mask)

        # Target embedding + positional encoding
        dec_inp_embed =  self.embedding_lookup(target_ids )
        dec_inp_embed = dec_inp_embed + self.position_embedding(target_ids)
        #
        dec_out = self.decoder( sequence_output, decode_attention_mask,dec_inp_embed, target_attention_mask)
        #
        # # Make the prediction out of the decoder output.
        logits = self.dense(dec_out)  # [batch, target_vocab]

        return  logits


class SimpleTransformer2Heads(tf.keras.layers.Layer):
    def __init__(self, config, float_type=tf.float32, **kwargs):
        super(SimpleTransformer2Heads, self).__init__(**kwargs)
        self.config =  config
        self.float_type = float_type

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.embedding_lookup = EmbeddingLookup(
            vocab_size=self.config.vocab_size,
            embedding_size=self.config.hidden_size,
            initializer_range=self.config.initializer_range,
            dtype=tf.float32,
            name="word_embeddings")
        self.embedding_postprocessor = EmbeddingPostprocessor(
            use_type_embeddings=True,
            token_type_vocab_size=  self.config.type_vocab_size,
            use_position_embeddings=True,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_prob=self.config.hidden_dropout_prob,
            initializer_range=self.config.initializer_range,
            dtype=tf.float32,
            name="embedding_postprocessor")
        self.encoder = bert_modeling.Transformer(
            num_hidden_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_activation=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            backward_compatible=self.config.backward_compatible,
            float_type=self.float_type,
            name="encoder")

        self.decoder =  DecodeTransformer(
            num_hidden_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_activation=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            backward_compatible=self.config.backward_compatible,
            float_type=self.float_type,
            name="decoder")

        self.pooler_transform = tf.keras.layers.Dense(
            units=self.config.hidden_size,
            activation="tanh",
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="pooler_transform")

        self.dense = tf.keras.layers.Dense(self.config.vocab_size)

        self.position_embedding = PositionalEmbeddingLookup(self.config.max_answer_length)

        super(SimpleTransformer2Heads, self).build(unused_input_shapes)

    def __call__(self,
                 input_word_ids,
                 input_mask=None,
                 input_type_ids=None,
                 target_ids = None,
                 target_mask = None,
                 **kwargs):
        if type(input_word_ids) is tuple:
            inputs = input_word_ids
        else:
            inputs = (input_word_ids, input_mask, input_type_ids, target_ids,target_mask)

        #inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids,target_ids,target_mask])
        return super(SimpleTransformer2Heads, self).__call__(inputs, **kwargs)

    def call(self, inputs ):
        """Implements call() for the layer.

        Args:
          inputs: packed input tensors.
          mode: string, `coqa` or `encoder`.
        Returns:

        """
        #unpacked_inputs = tf_utils.unpack_inputs(inputs)
        # input_word_ids = unpacked_inputs[0]
        # input_mask = unpacked_inputs[1]
        # input_type_ids = unpacked_inputs[2]
        # target_ids = unpacked_inputs[3]
        # target_mask =  unpacked_inputs[4]

        (input_word_ids, input_mask, input_type_ids, target_ids, target_mask) =inputs
        word_embeddings = self.embedding_lookup(input_word_ids)
        embedding_tensor = self.embedding_postprocessor(
            word_embeddings=word_embeddings, token_type_ids=input_type_ids)
        if self.float_type == tf.float16:
            embedding_tensor = tf.cast(embedding_tensor, tf.float16)
        attention_mask = None
        if input_mask is not None:
            attention_mask = bert_modeling.create_attention_mask_from_input_mask(
                input_word_ids, input_mask)

        target_attention_mask =None
        if target_mask is not None:
            target_attention_mask = bert_modeling.create_attention_mask_from_input_mask(
                target_ids, target_mask)
            target_attention_mask *= construct_autoregressive_mask(target_ids)

        sequence_output = self.encoder(embedding_tensor, attention_mask)

        if input_mask is not None:
            decode_attention_mask = bert_modeling.create_attention_mask_from_input_mask(
                target_ids, input_mask)

        # Target embedding + positional encoding
        dec_inp_embed =  self.embedding_lookup(target_ids )
        dec_inp_embed = dec_inp_embed + self.position_embedding(target_ids)
        #
        dec_out = self.decoder( sequence_output, decode_attention_mask,dec_inp_embed, target_attention_mask)
        #
        # # Make the prediction out of the decoder output.
        logits = self.dense(dec_out)  # [batch, target_vocab]

        return  logits, sequence_output


class DecodeTransformerBlock(tf.keras.layers.Layer):
  """Single transformer layer.

  It has two sub-layers. The first is a multi-head self-attention mechanism, and
  the second is a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.1,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(DecodeTransformerBlock, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (self.hidden_size, self.num_attention_heads))
    self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.attention_layer = bert_modeling.Attention(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.attention_head_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_range=self.initializer_range,
        backward_compatible=self.backward_compatible,
        name="self_attention")
    self.decode_attention_layer = bert_modeling.Attention(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.attention_head_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_range=self.initializer_range,
        backward_compatible=self.backward_compatible,
        name="decode_attention_layer")

    self.attention_output_dense = bert_modeling.Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=int(self.hidden_size / self.num_attention_heads),
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=True,
        backward_compatible=self.backward_compatible,
        name="self_attention_output")
    self.decode_attention_output_dense = bert_modeling.Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=int(self.hidden_size / self.num_attention_heads),
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=True,
        backward_compatible=self.backward_compatible,
        name="decode_self_attention_output")
    self.attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
            # We do layer norm in float32 for numeric stability.
            dtype=tf.float32))
    self.intermediate_dense = bert_modeling.Dense2DProjection(
        output_size=self.intermediate_size,
        kernel_initializer=get_initializer(self.initializer_range),
        activation=self.intermediate_activation,
        # Uses float32 so that gelu activation is done in float32.
        fp32_activation=True,
        name="intermediate")
    self.output_dense = bert_modeling.Dense2DProjection(
        output_size=self.hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        name="output")
    self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    super(DecodeTransformerBlock, self).build(unused_input_shapes)

  def common_layers(self):
    """Explicitly gets all layer objects inside a Transformer encoder block."""
    return [
        self.attention_layer, self.attention_output_dense,
        self.attention_dropout, self.attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_dropout,
        self.output_layer_norm
    ]

  def __call__(self, target_tensor, input_tensor, attention_mask,target_mask, **kwargs):
    inputs = tf_utils.pack_inputs([target_tensor,input_tensor, attention_mask,target_mask])

    if type(target_tensor) is tuple:
        inputs = target_tensor
    else:
        inputs = (target_tensor, input_tensor, attention_mask, target_mask)

    return super(DecodeTransformerBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    (target_tensor, input_tensor, attention_mask, target_mask) = inputs

    #(target_tensor,input_tensor, attention_mask,target_mask) = tf_utils.unpack_inputs(inputs)

    output = self.decode_attention_layer(
        from_tensor=target_tensor,
        to_tensor=target_tensor,
        attention_mask=target_mask)

    output = self.decode_attention_output_dense(output)
    output = self.attention_layer_norm(target_tensor + output)

    attention_output = self.attention_layer(
        from_tensor=output,
        to_tensor=input_tensor,
        attention_mask=attention_mask)

    attention_output = self.attention_output_dense(attention_output)
    attention_output = self.attention_layer_norm(attention_output + output)

    if self.float_type == tf.float16:
      attention_output = tf.cast(attention_output, tf.float16)

    intermediate_output = self.intermediate_dense(attention_output)

    if self.float_type == tf.float16:
      intermediate_output = tf.cast(intermediate_output, tf.float16)

    layer_output = self.output_dense(intermediate_output)

    # Use float32 in keras layer norm for numeric stability
    layer_output = self.output_layer_norm(layer_output + attention_output)
    if self.float_type == tf.float16:
      layer_output = tf.cast(layer_output, tf.float16)
    return layer_output


class DecodeTransformer(tf.keras.layers.Layer):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  """

  def __init__(self,
               num_hidden_layers=12,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(DecodeTransformer, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          DecodeTransformerBlock(
              hidden_size=self.hidden_size,
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              hidden_dropout_prob=self.hidden_dropout_prob,
              attention_probs_dropout_prob=self.attention_probs_dropout_prob,
              initializer_range=self.initializer_range,
              backward_compatible=self.backward_compatible,
              float_type=self.float_type,
              name=("layer_%d" % i)))
    super(DecodeTransformer, self).build(unused_input_shapes)

  def __call__(self, input_tensor, attention_mask,target_tensor,target_mask, **kwargs):

    if type(input_tensor) is tuple:
        inputs=input_tensor
    else:
        inputs =(input_tensor, attention_mask,target_tensor,target_mask)
    return super(DecodeTransformer, self).__call__(inputs=inputs, **kwargs)

  def call(self, inputs, return_all_layers=False):
    """Implements call() for the layer.

    Args:
      inputs: packed inputs.
      return_all_layers: bool, whether to return outputs of all layers inside
        encoders.
    Returns:
      Output tensor of the last layer or a list of output tensors.
    """
    # unpacked_inputs = tf_utils.unpack_inputs(inputs)
    # input_tensor = unpacked_inputs[0]
    # attention_mask = unpacked_inputs[1]
    # target_tensor =  unpacked_inputs[2]
    # target_mask =  unpacked_inputs[3]

    (input_tensor, attention_mask, target_tensor, target_mask) =inputs
    output_tensor = target_tensor

    all_layer_outputs = []
    for layer in self.layers:
      output_tensor = layer(output_tensor,input_tensor,attention_mask,target_mask)
      all_layer_outputs.append(output_tensor)

    if return_all_layers:
      return all_layer_outputs

    return all_layer_outputs[-1]

def construct_autoregressive_mask(target):
    """
    Args: Original target of word ids, shape [batch, seq_len]
    Returns: a mask of shape [batch, seq_len, seq_len].
    """
    batch_size = tf_utils.get_shape_list(target)[0]
    seq_len = tf_utils.get_shape_list(target)[1]

    tri_matrix = tf.ones((seq_len, seq_len))
    operator = tf.linalg.LinearOperatorLowerTriangular(tri_matrix)

    mask = operator.to_dense()
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(mask, 0),[batch_size, 1, 1])
    masks = tf.cast(masks, dtype = tf.int32)
    return masks