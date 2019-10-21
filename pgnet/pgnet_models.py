

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import tensorflow_hub as hub

from modeling import tf_utils
import pgnet.pgnet_modeling as modeling
import bert.bert_modeling as bert_modeling



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
def coqa_model(bert_config, max_seq_length,max_answer_length, float_type, initializer=None):
  """Returns BERT Coqa model along with core BERT model to import weights.

  Args:
    bert_config: BertConfig, the config defines the core Bert model.
    max_seq_length: integer, the maximum input sequence length.
    float_type: tf.dtype, tf.float32 or tf.bfloat16.
    initializer: Initializer for weights in BertSquadLogitsLayer.

  Returns:
    Two tensors, start logits and end logits, [batch x sequence length].
  """
  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_ids')
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

  coqa= modeling.get_pgnet_model(
                                 input_states=sequence_output,
                                 answer_ids=answer_ids,
                                 answer_mask=answer_mask,
                                 config=bert_config)

  return coqa, core_model

def train_model(pgnet_config,
                   seq_length,
                   max_predictions_per_seq,
                   initializer=None):
  """Returns model to be used for training.

  Args:
      pgnet_config: Configuration that defines the core BERT model.
      seq_length: Maximum sequence length of the training data.
      max_predictions_per_seq: Maximum number of tokens in sequence to mask out
        and use for pretraining.
      initializer: Initializer for weights in BertPretrainLayer.

  Returns:
      Pretraining model as well as core BERT submodel from which to save
      weights after pretraining.
  """

  input_word_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
  input_mask = tf.keras.layers.Input(
      shape=(seq_length,), name='input_mask', dtype=tf.int32)
  input_type_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_type_ids', dtype=tf.int32)
    
  answer_word_ids = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)

  pgnet_submodel_name = 'pgnet_model'
  pgnet_submodel = modeling.get_pgnet_model(
      input_word_ids,
      input_mask,
      input_type_ids,
      name=pgnet_submodel_name,
      config=pgnet_config)

  finalDist = pgnet_submodel.outputs[0] 


  pretrain_loss_layer = PGNetTrainLossAndMetricLayer(pgnet_config)
  output_loss = pretrain_loss_layer( finalDist)

  return tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids,
          'answer_word_ids': answer_word_ids
      },
      outputs=output_loss), pgnet_submodel

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



class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, vocab):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._vocab = vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess)

    if FLAGS.single_pass:
      # Make a descriptive decode directory name
      ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
      self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
      if os.path.exists(self._decode_dir):
        raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

    else: # Generic decode dir name
      self._decode_dir = os.path.join(FLAGS.log_root, "decode")

    # Make the decode dir if necessary
    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

    if FLAGS.single_pass:
      # Make the dirs to contain output written in the correct format for pyrouge
      self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
      if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
      self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
      if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)


  def decode(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)
        return

      original_article = batch.original_articles[0]  # string
      original_abstract = batch.original_abstracts[0]  # string
      original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

      article_withunks = data.show_art_oovs(original_article, self._vocab) # string
      abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

      # Run beam search to get best Hypothesis
      best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

      # Extract the output ids from the hypothesis and convert back to words
      output_ids = [int(t) for t in best_hyp.tokens[1:]]
      decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

      # Remove the [STOP] token from decoded_words, if necessary
      try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
      except ValueError:
        decoded_words = decoded_words
      decoded_output = ' '.join(decoded_words) # single string

      if FLAGS.single_pass:
        self.write_for_rouge(original_abstract_sents, decoded_words, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
        counter += 1 # this is how many examples we've decoded
      else:
        print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
        self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool

        # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
        t1 = time.time()
        if t1-t0 > SECS_UNTIL_NEW_CKPT:
          tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
          _ = util.load_ckpt(self._saver, self._sess)
          t0 = time.time()

class PGNetTrainLossAndMetricLayer(tf.keras.layers.Layer):
    """Returns layer that computes custom loss and metrics for pretraining."""

    def __init__(self, pgnet_config, **kwargs):
        super(PGNetTrainLossAndMetricLayer, self).__init__(**kwargs)
        self.config = copy.deepcopy(pgnet_config)


    def __call__(self,
                 lm_output,
                 sentence_output=None,
                 lm_label_ids=None,
                 lm_label_weights=None,
                 sentence_labels=None):
        inputs = tf_utils.pack_inputs([
            lm_output, sentence_output, lm_label_ids, lm_label_weights,
            sentence_labels
        ])
        return super(PGNetTrainLossAndMetricLayer, self).__call__(inputs)

    def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                     lm_per_example_loss, sentence_output, sentence_labels,
                     sentence_per_example_loss):
        """Adds metrics."""
          
    def call(self, inputs):
        """Implements call() for the layer."""
        unpacked_inputs = tf_utils.unpack_inputs(inputs)

        if self.config.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
            batch_nums = tf.range(0, limit=self.config.batch_size)  # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
                targets = self._target_batch[:, dec_step]  # The indices of the target words. shape (batch_size)
                indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
                losses = -tf.log(gold_probs)
                loss_per_step.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

        else:  # baseline model
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch,
                                                          self._dec_padding_mask)  # this applies softmax internally

        tf.summary.scalar('loss', self._loss)

        # Calculate coverage loss from the attention distributions
        if hps.coverage:
            with tf.variable_scope('coverage_loss'):
                self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)



def run_beam_search(sess, model, vocab, batch):
  """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  # Run the encoder to get the encoder hidden states and decoder initial state
  enc_states, dec_in_state = model.run_encoder(sess, batch)
  # dec_in_state is a LSTMStateTuple
  # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                     log_probs=[0.0],
                     state=dec_in_state,
                     attn_dists=[],
                     p_gens=[],
                     coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length
                     ) for _ in xrange(FLAGS.beam_size)]
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)

  steps = 0
  while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
    latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
    latest_tokens = [t if t in xrange(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
    states = [h.state for h in hyps] # list of current decoder states of the hypotheses
    prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)

    # Run one step of the decoder to get the new info
    (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                        batch=batch,
                        latest_tokens=latest_tokens,
                        enc_states=enc_states,
                        dec_init_states=states,
                        prev_coverage=prev_coverage)

    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    for i in xrange(num_orig_hyps):
      h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i]  # take the ith hypothesis and new decoder state info
      for j in xrange(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
        # Extend the ith hypothesis with the jth option
        new_hyp = h.extend(token=topk_ids[i, j],
                           log_prob=topk_log_probs[i, j],
                           state=new_state,
                           attn_dist=attn_dist,
                           p_gen=p_gen,
                           coverage=new_coverage_i)
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
    for h in sort_hyps(all_hyps): # in order of most likely h
      if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= FLAGS.min_dec_steps:
          results.append(h)
      else: # hasn't reached stop token, so continue to extend this hypothesis
        hyps.append(h)
      if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1

  # At this point, either we've got beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results)

  # Return the hypothesis with highest average log prob
  return hyps_sorted[0]



def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)