

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import tensorflow_hub as hub

from modeling import tf_utils
import pgnet.pgnet_modeling as modeling
import bert.bert_modeling as bert_modeling
import bert.bert_models as bert_models


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



class PGNetTrainLayer(tf.keras.layers.Layer):

  def __init__(self,
               config,
               pgnet_layer,
               initializer=None,
               float_type=tf.float32,
               **kwargs):
    super(PGNetTrainLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)
    self.float_type = float_type

    self.embedding_table = pgnet_layer.embedding_lookup.embeddings
    self.num_next_sentence_label = 2
    if initializer:
      self.initializer = initializer
    else:
      self.initializer = tf.keras.initializers.TruncatedNormal(
          stddev=self.config.initializer_range)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""

    
    super(PGNetTrainLayer, self).build(unused_input_shapes)

  def __call__(self,
               pooled_output,
               sequence_output=None,
               masked_lm_positions=None):
    inputs = tf_utils.pack_inputs(
        [pooled_output, sequence_output, masked_lm_positions])
    return super(PGNetTrainLayer, self).__call__(inputs)

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)

    return

# class CoqaModel(tf.keras.Model):
#
#   def __init__(self,bert_config,float_type,**kwargs):
#       super(CoqaModel, self).__init__(**kwargs)
#
#       self.pgnet_model_layer = modeling.PGNetSummaryModel(config=bert_config,
#                                                      float_type=float_type,
#                                                      name='pgnet_summary_model')
#
#   def call(self, inputs,**kwargs):
#       print(inputs)
#       input_ids, input_mask, answer_ids, answer_mask=inputs
#       final_dists, attn_dists = self.pgnet_model_layer(input_ids=input_ids,
#                                                   input_mask=input_mask,
#                                                   answer_ids=answer_ids,
#                                                   answer_mask=answer_mask
#                                                   )
#       return final_dists, attn_dists


def coqa_model(bert_config, max_seq_length,max_answer_length,max_oov_size, float_type,training = False, initializer=None):
  """Returns BERT Coqa model along with core BERT model to import weights.

  Args:
    bert_config: BertConfig, the config defines the core Bert model.
    max_seq_length: integer, the maximum input sequence length.
    float_type: tf.dtype, tf.float32 or tf.bfloat16.
    initializer: Initializer for weights in BertSquadLogitsLayer.

  Returns:
    Two tensors, start logits and end logits, [batch x sequence length].
  """
  unique_ids = tf.keras.layers.Input(
      shape=(1,), dtype=tf.int32, name='unique_ids')
  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')
  answer_ids = tf.keras.layers.Input(
      shape=(max_answer_length,), dtype=tf.int32, name='answer_ids')
  answer_mask = tf.keras.layers.Input(
      shape=(max_answer_length,), dtype=tf.int32, name='answer_mask')

  core_model = bert_modeling.get_bert_model(
      input_word_ids,
      input_mask,
      input_type_ids,
      config=bert_config,
      name='bert_model',
      float_type=float_type)

  # `Bert Coqa Pgnet Model` only uses the sequnce_output which
  # has dimensionality (batch_size, sequence_length, num_hidden).
  sequence_output = core_model.outputs[1]


  if initializer is None:
        initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

  #
  # Double headed- trained on both span positions and final answer
  #
  # coqa_logits_layer = bert_models.BertCoqaLogitsLayer(
  #     initializer=initializer, float_type=float_type, name='coqa_logits')
  # start_logits, end_logits = coqa_logits_layer(sequence_output)
  #
  # #figure out the span text from the start logits and end_logits here.
  # span_text_ids,span_mask=get_best_span_prediction(input_word_ids, start_logits, end_logits,max_seq_length )
  #
  # pgnet_model_layer =modeling.PGNetSummaryModel(config=bert_config ,
  #                                                 float_type=float_type,
  #                                                name='pgnet_summary_model')
  # final_dists, attn_dists = pgnet_model_layer(  span_text_ids,
  #                                               span_mask,
  #                                               answer_ids,
  #                                               answer_mask
  #                                             )
  # coqa = tf.keras.Model(
  #     inputs=[
  #         unique_ids,
  #         answer_ids,
  #         answer_mask ],
  #     outputs=[final_dists, attn_dists,start_logits, end_logits ])

  #PGNet only: end to end - question+context to answer

  coqa_layer = CoqaModel(config=bert_config ,training=training,
                                                  float_type=float_type )

  final_dists  = coqa_layer(  input_word_ids,
                                                input_mask,
                                                answer_ids,
                                                answer_mask,
                                              )



  coqa = tf.keras.Model(
      inputs=({
         'unique_ids': unique_ids,
         'input_word_ids': input_word_ids,
         'input_mask': input_mask,
         'answer_ids': answer_ids,
         'answer_mask': answer_mask,
         'input_type_ids': input_type_ids},),

      outputs=[unique_ids,final_dists ])

  coqa.add_loss(coqa_layer.losses)
  

  # Bert+PGNet:  end to end
  # pgnet_model_layer = modeling.PGNetDecoderModel(config=bert_config ,
  #                                                  float_type=float_type,
  #                                                  name='pgnet_decoder_model')
  # final_dists, attn_dists = pgnet_model_layer(sequence_output, answer_ids, answer_mask)
  # coqa = tf.keras.Model(
  #     inputs=[
  #         unique_ids,
  #         answer_ids,
  #         answer_mask],
  #     outputs=[unique_ids, final_dists, attn_dists],
  #     name="pgnet_model")

  return coqa, core_model


def get_best_span_prediction(ids, start_logits, end_logits,max_len):
    _, starts = tf.nn.top_k(start_logits, k=1)
    _, ends = tf.nn.top_k(end_logits, k=1)

    s = tf.transpose(tf.tile(starts, [1, max_len]))
    e = tf.transpose(tf.tile(ends, [1, max_len]))

    ta = tf.TensorArray(dtype=tf.int32, size=max_len)
    # print(s,e)
    for i in tf.range(max_len):
        x = tf.where(i >= s[i], 1, 0)
        y = tf.where(i < e[i], 1, 0)
        # tf.print(x,y)
        ta.write(i, x * y)

    m = tf.transpose(ta.stack(), [1, 0])
    spans = ids * m
    # print(spans,starts)

    # for i in tf.range(max_len):
    # new_spans=tf.roll(spans, shift=s[i], axis=[1])
    # def f_roll(arg):
    #     x, s = arg
    #     return tf.roll(x, shift=-1 * s, axis=[0])
    #     # tf.roll(x, shift= -1*s , axis=[0])
    #
    # new_spans = tf.vectorized_map(
    #     fn=f_roll,
    #     elems=(spans, starts)
    # )
    #
    # new_mask = tf.vectorized_map(
    #     fn=f_roll,
    #     elems=(m, starts)
    # )
    #return (new_spans, new_mask)
    return (spans, m)

class CoqaModel(tf.keras.layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 config,
                 training=False,
                 float_type=tf.float32,
                 name = 'coqa_model',
                 **kwargs):
        super(CoqaModel, self).__init__(name=name, **kwargs)

        self.config = (
            modeling.PGNetConfig.from_dict(config)
            if isinstance(config, dict) else copy.deepcopy(config))

        self.float_type = float_type
        self.training = training

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.embedding_lookup = modeling.EmbeddingLookup(self.config.vocab_size,
                                                self.config.hidden_size,
                                                dtype=tf.float32,
                                                )
        self.encoder = modeling.Encoder(self.config.hidden_size, self.config.max_seq_length, dynamic=True)
        self.decoder = modeling.AttentionDecoder(self.config.hidden_size, self.config.hidden_size,
                                        self.config.max_seq_length, modeling.get_initializer())
        self.output_projector = modeling.OutputProjectionLayer(self.config.hidden_size, self.config.vocab_size)
        self.final_distribution = modeling.FinalDistributionLayer(self.config.hidden_size, self.config.vocab_size,
                                                         self.config.max_oov_size)

        super(CoqaModel, self).build(unused_input_shapes)

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
        return super(CoqaModel, self).__call__(inputs, **kwargs)
    def call(self,inputs):
        input_ids, input_mask, answer_ids, answer_mask = inputs
        emb_enc_inputs = self.embedding_lookup(input_ids)
        # tensor with shape (batch_size, max_seq_length, emb_size)

        emb_dec_inputs = [self.embedding_lookup(x) for x in tf.unstack(answer_ids, axis=1)]
        # list length max_dec_steps containing shape (batch_size, emb_size)
        if not self.training:
            emb_dec_inputs=emb_dec_inputs[:1]
            #we only have the [START] token



        enc_outputs, enc_state = self.encoder(emb_enc_inputs, input_mask)

        _enc_states = enc_outputs

        _dec_in_state = enc_state

        atten_len = tf_utils.get_shape_list(input_ids)[1]
        batch_size = tf_utils.get_shape_list(input_ids)[0]

        prev_coverage = None  # self.prev_coverage #if self.config.mode == "decode" and self.config.use_coverage  else None

        if self.training:
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

            def _mask_and_avg(values, padding_mask):
                """Applies mask to values then returns overall average (a scalar)

                Args:
                  values: a list length max_dec_steps containing arrays shape (batch_size).
                  padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

                Returns:
                  a scalar
                """
                padding_mask = tf.cast(padding_mask, tf.dtypes.float32)
                dec_lens = (tf.reduce_sum(padding_mask, axis=1))  # shape batch_size. float32
                values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
                values_per_ex = sum(
                    values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
                return tf.reduce_mean(values_per_ex)  # overall average

            def _coverage_loss(attn_dists, padding_mask):
                """Calculates the coverage loss from the attention distributions.

                Args:
                  attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
                  padding_mask: shape (batch_size, max_dec_steps).

                Returns:
                  coverage_loss: scalar
                """
                coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
                covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
                for a in attn_dists:
                    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
                    covlosses.append(covloss)
                    coverage += a  # update the coverage vector
                coverage_loss = _mask_and_avg(covlosses, padding_mask)
                return coverage_loss

            #add the coverage loss
            self.add_loss(_coverage_loss(attn_dists,answer_mask))

            # the main loss will be based on predictions, we will let the training loop handle it.

            return final_dists
        else:

            def sort_hyps(hyps):
                """Return a list of Hypothesis objects, sorted by descending average log probability"""
                return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)

            # we do a beam search on step by step decoding
            UNKNOWN_TOKEN = 0
            START_TOKEN = 104
            STOP_TOKEN = 105



            hyps = [{"tokens":[START_TOKEN],  # answer_ids[0] is the [START]
                               "log_probs":[0.0],
                               "state":_dec_in_state,
                               "attn_dists":[],
                               "p_gens":[],
                               "coverage":tf.zeros(atten_len)  # zero vector of length attention_length
                     } ] *  batch_size

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
                num_orig_hyps = 1 if steps == 0 else len(hyps)
                # On the first step, we only had one original hypothesis (the initial hypothesis).
                # On subsequent steps, all original hypotheses are distinct.

                for i in tf.range(num_orig_hyps):
                    h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                                     new_coverage[i]
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


    def get_config(self):
        config = {"config": self.config.to_dict()}
        base_config = super(CoqaModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

