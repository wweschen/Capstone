from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import tensorflow as tf

from modeling import tf_utils

def get_initializer(initializer_range=0.02):
  """Creates a `tf.initializers.truncated_normal` with the given range.

  Args:
    initializer_range: float, initializer range for stddev.

  Returns:
    TruncatedNormal initializer with stddev = `initializer_range`.
  """
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

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

def get_pgnet_model(

                   input_states,
                   answer_ids,
                   answer_mask,
                   config=None,
                   name=None,
                   float_type=tf.float32):
  """Wraps the core PGNetSummaryModel model as a keras.Model."""
  pgnet_model_layer = PGNetDecoderModel(config=config, float_type=float_type, name=name)
  final_dists,attn_dists = pgnet_model_layer(input_states, answer_ids, answer_mask )
  pgnet_model = tf.keras.Model(
      inputs=[
              input_states,
              answer_ids,
              answer_mask],
      outputs=[final_dists,attn_dists])

  return pgnet_model

class PGNetDecoderModel(tf.keras.layers.Layer):
    def __init__(self,
                 config,
                 float_type=tf.float32,
                 **kwargs):
        super(PGNetDecoderModel, self).__init__(**kwargs)

        self.config = (
            PGNetConfig.from_dict(config)
            if isinstance(config, dict) else copy.deepcopy(config))

        self.float_type = float_type

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""

        self.embedding_lookup = EmbeddingLookup(self.config.vocab_size, self.config.hidden_size)

        self.state_reducer = ReduceStateLayer(self.config.hidden_size)
        self.decoder = AttentionDecoder(self.config.hidden_size, self.config.vocab_size,
                                        self.config.max_answer_length, get_initializer())
        self.output_projector = OutputProjectionLayer(self.config.hidden_size, self.config.vocab_size)
        self.final_distribution = FinalDistributionLayer(self.config.hidden_size, self.config.vocab_size)

        super(PGNetDecoderModel, self).build(unused_input_shapes)

    def __call__(self,
                 enc_states,
                 answer_ids,
                 answer_mask,
                 **kwargs):
        inputs = tf_utils.pack_inputs([enc_states, answer_ids, answer_mask])
        return super(PGNetDecoderModel, self).__call__(inputs, **kwargs)

    def call(self, inputs, mode="pgnet"):
        unpacked_inputs = tf_utils.unpack_inputs(inputs)

        enc_states = unpacked_inputs[0]
        answer_ids = unpacked_inputs[1]
        dec_mask = unpacked_inputs[2]

        emb_dec_inputs = [self.embedding_lookup(x) for x in tf.unstack(answer_ids,
                                                                       axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)

        self._enc_states = enc_states

        self._dec_in_state = None

        #prev_coverage = self.prev_coverage if self.config.mode == "decode" and self.config.use_coverage else None

        decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self.decoder(
            emb_dec_inputs,
            self._dec_in_state,
            self._enc_states,
            dec_mask )

        vocab_dists = self.output_projector(decoder_outputs)

        if self.config.use_pointer_gen:
            final_dists = self.final_distribution(vocab_dists, self.attn_dists)
        else:  # final distribution is just vocabulary distribution
            final_dists = vocab_dists

        return final_dists,self.attn_dists

    def get_config(self):
        config = {"config": self.config.to_dict()}
        base_config = super(PGNetDecoderModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PGNetSummaryModel(tf.keras.layers.Layer):
  def __init__(self,
               config,
               float_type=tf.float32,
               **kwargs):
    super(PGNetSummaryModel, self).__init__(**kwargs)

    self.config = (
        PGNetConfig.from_dict(config)
        if isinstance(config, dict) else copy.deepcopy(config))

    self.float_type = float_type



  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embedding_lookup = EmbeddingLookup(self.config.vocab_size,self.config.emb_dim)
    self.encoder = Encoder(self.config.emb_dim)
    self.state_reducer = ReduceStateLayer(self.config.hidden_size)
    self.decoder = AttentionDecoder(self.config.hidden_size,self.config.vocab_size,
                                    self.config.max_enc_steps,get_initializer())
    self.output_projector = OutputProjectionLayer(self.config.hidden_size,self.config.vocab_size)
    self.final_distribution = FinalDistributionLayer(self.config.hidden_size,self.config.vocab_size)

    super(PGNetSummaryModel, self).build(unused_input_shapes)

  def __call__(self,
               input_word_ids,
               input_mask=None,
               output_answer_ids=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([input_word_ids, input_mask, output_answer_ids])
    return super(PGNetSummaryModel, self).__call__(inputs, **kwargs)

  def call(self, inputs,mode="pgnet"):
      unpacked_inputs = tf_utils.unpack_inputs(inputs)
      input_word_ids = unpacked_inputs[0]
      input_mask = unpacked_inputs[1]
      output_answer_ids= unpacked_inputs[2]

      emb_enc_inputs = self.embedding_lookup(input_word_ids)  # tensor with shape (batch_size, max_enc_steps, emb_size)
      emb_dec_inputs = [self.embedding_lookup(x) for x in tf.unstack(output_answer_ids, axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)
      enc_outputs, fw_st, bw_st = self.encoder(emb_enc_inputs, self._enc_lens)


      self._enc_states = enc_outputs

      self._dec_in_state = self.state_reducer(fw_st, bw_st)

      if mode=="encoder":
         return (self._enc_states,self._dec_in_state )

      prev_coverage = self.prev_coverage if self.config.mode == "decode" and self.config.use_coverage  else None

      decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self.decoder(
                emb_dec_inputs,
                 self._dec_in_state ,
                 self._enc_states ,
                 input_mask,
                 prev_coverage=prev_coverage)
      if mode == "decoder":
           return (decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage)

      vocab_dists=self.output_projector(decoder_outputs)

      if self.config.use_pointer_gen:
         final_dists = self.final_distribution(vocab_dists, self.attn_dists)
      else:  # final distribution is just vocabulary distribution
          final_dists = vocab_dists

      return  final_dists

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


class Encoder (tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim
                 ):
        self.hidden_dim=hidden_dim

    def build(self, unused_input_shapes):

        self.lstm_layer = tf.keras.layers.LSTMCell(self.hidden_dim,return_sequences=True )

        self.bidirectional_layer=tf.keras.layers.Bidirectional(self.lstm_layer)

        super(Encoder, self).build(unused_input_shapes)

    def __call__(self,
                 encoder_inputs, seq_len,
                 **kwargs):
        inputs = tf_utils.pack_inputs([encoder_inputs, seq_len])

        return super(Encoder, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        encoder_inputs = unpacked_inputs[0]
        seq_len = unpacked_inputs[1]

        (encoder_outputs, (fw_st, bw_st)) = self.bidirectional_layer(  encoder_inputs)

        encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states


        return encoder_outputs, fw_st, bw_st

class ReduceStateLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim
                 ):
        self.hidden_dim=hidden_dim

    def build(self, unused_input_shapes):
        hidden_dim = self.hidden_dim
        self.w_reduce_c = self.add_weight('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                     initializer=self.trunc_norm_init)
        self.w_reduce_h = self.add_weight('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                     initializer=self.trunc_norm_init)
        self.bias_reduce_c = self.add_weight('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                        initializer=self.trunc_norm_init)
        self.bias_reduce_h = self.add_weight('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                        initializer=self.trunc_norm_init)
        super(ReduceStateLayer, self).build(unused_input_shapes)

    def __call__(self,
                 fw_state, bw_state,
                 **kwargs):
        inputs = tf_utils.pack_inputs([fw_state, bw_state])

        return super(ReduceStateLayer, self).__call__(inputs, **kwargs)
    def call(self, inputs):
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        fw_state = unpacked_inputs[0]
        bw_state = unpacked_inputs[1]

        # Apply linear layer
        old_c = tf.concat(axis=1, values=[fw_state.c, bw_state.c])  # Concatenation of fw and bw cell
        old_h = tf.concat(axis=1, values=[fw_state.h, bw_state.h])  # Concatenation of fw and bw state
        new_c = tf.nn.relu(tf.matmul( old_c, self.w_reduce_c) + self.bias_reduce_c)  # Get new cell from old cell
        new_h = tf.nn.relu(tf.matmul( old_h, self.w_reduce_h) + self.bias_reduce_h)  # Get new state from old state
        return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_c, new_h)  # Return new cell and state


class OutputProjectionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 vocab_size
                 ):
        self.hidden_dim=hidden_dim
        self.vocab_size =vocab_size


    def build(self, unused_input_shapes):
        w = self.add_weight('w', [self.hidden_dim, self.vocab_size], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [self.vocab_size], dtype=tf.float32, initializer=self.trunc_norm_init)
        super(OutputProjectionLayer, self).build(unused_input_shapes)


    def call(self, inputs):
        decoder_outputs = inputs

        vocab_scores = []  # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i, output in enumerate(decoder_outputs):
            vocab_scores.append(tf.nn.xw_plus_b(output, self.w, self.v))  # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for s in
                       vocab_scores]
        # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

        return vocab_dists


class FinalDistributionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 vocab_size
                 ):

        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size

    #def build(self, unused_input_shapes):


    def __call__(self,
                 vocab_dists, attn_dists,p_gens,
                 **kwargs):
        inputs = tf_utils.pack_inputs([vocab_dists, attn_dists,p_gens])

        return super(FinalDistributionLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        vocab_dists=unpacked_inputs[0]
        attn_dists=unpacked_inputs[1]
        p_gens=unpacked_inputs[2]

        vocab_dists = [p_gen * dist for (p_gen, dist) in zip( p_gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip( p_gens, attn_dists)]

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        extended_vsize = self.vocab_size + self._max_art_oovs  # the maximum (over the batch) size of the extended vocabulary
        extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
                                vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=self._hps.batch_size)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
        indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
        shape = [self._hps.batch_size, extended_vsize]
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
                                attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                       zip(vocab_dists_extended, attn_dists_projected)]

        return final_dists

class AttentionDecoder (tf.keras.layers.Layer):
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
        super(AttentionDecoder , self).__init__(**kwargs)
        self.initializer = initializer
        self.float_type = float_type
        self.hidden_dim=hidden_dim
        self.vector_size = vector_size
        self.attention_length = attention_length
        self.pointer_gen = pointer_gen
        self.use_coverage = use_coverage
        self.initial_state_attention=initial_state_attention

    def build(self, unused_input_shapes):
        self.lstm_layer = tf.keras.layers.LSTMCell(self.hidden_dim)
        self.attention_layer = AttentionLayer(self.vector_size,self.attention_length,True)



    def __call__(self,
                 decoder_inputs,
                 dec_initial_state,
                 encoder_states,
                 enc_padding_mask,
                 prev_coverage=None,
                 **kwargs):
        inputs = tf_utils.pack_inputs([decoder_inputs,  dec_initial_state,encoder_states,enc_padding_mask,prev_coverage])
        return super(AttentionDecoder, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        unpacked_inputs = tf_utils.unpack_inputs(inputs)

        decoder_inputs = unpacked_inputs[0]
        initial_state = unpacked_inputs[1]
        encoder_states = unpacked_inputs[2]
        enc_padding_mask = unpacked_inputs[3]
        prev_coverage =  unpacked_inputs[4]

        input_shape = tf_utils.get_shape_list(encoder_states, expected_rank=[2, 3])
        batch_size = input_shape[0]

        outputs = []
        attn_dists = []
        p_gens = []
        state = initial_state
        coverage = prev_coverage  # initialize coverage to None or whatever was passed in
        context_vector = tf.zeros([batch_size, self.attention_length])
        context_vector.set_shape([None, self.attention_length])  # Ensure the second shape of attention vectors is set.
        if self.initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
            context_vector, _, coverage = self.attention_layer ( encoder_state=encoder_states,
                                                                 decoder_state=initial_state,
                                                                 coverage =coverage,
                                                                 input_mask=enc_padding_mask )
            # in decode mode, this is what updates the coverage vector

        for i, inp in enumerate(decoder_inputs):

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            x = tf.keras.layers.Dense(input_size)([inp] + [context_vector])

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = self.lstm_layer(x, state)

            # Run the attention mechanism.
            if i == 0 and self.initial_state_attention:  # always true in decode mode
                context_vector, attn_dist, _ = self.attention_layer (encoder_state=encoder_states,
                                                                     decoder_state=state,
                                                                     coverage=coverage,
                                                                           input_mask=enc_padding_mask)  # don't allow coverage to update
            else:
                context_vector, attn_dist, coverage = self.attention_layer(encoder_state=encoder_states,
                                                                           decoder_state=state,
                                                                           coverage=coverage,
                                                                           input_mask=enc_padding_mask)
            attn_dists.append(attn_dist)

            # Calculate p_gen
            if self.pointer_gen:
                p_gen = tf.keras.layers.Dense(1)( [context_vector, state.c, state.h, x])
                # Tensor shape (batch_size, 1)
                p_gen = tf.sigmoid(p_gen)
                p_gens.append(p_gen)

                # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
                # This is V[s_t, h*_t] + b in the paper
                output = tf.keras.layers.Dense(self.lstm_layer.output_size)( [cell_output] + [context_vector])
            outputs.append(output)

        # If using coverage, reshape it
        if coverage is not None:
            coverage = tf.reshape(coverage, [batch_size, -1])

        return outputs, state, attn_dists, p_gens, coverage


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, vector_size,attention_length, use_coverage,initializer=None, float_type=tf.float32, **kwargs):
        super(AttentionLayer , self).__init__(**kwargs)
        self.initializer = initializer
        self.float_type = float_type
        self.vector_size = vector_size
        self.attention_length=attention_length
        self.use_coverage=use_coverage

        self.w_h = self.add_weight(shape=[1, 1, self.attention_length, self.vector_size], name="W_h")
        self.v = self.add_weight(shape=[self.attention_length], name="v")
        self.w_c = self.add_weight(shape=[1, 1, 1, self.vector_size], name="W_c")

    def build(self,  unused_input_shapes):

        self.encoder_layer = tf.keras.layers.Conv2D(filters=self.attention_length,kernel_size= (1, self.vector_size), padding="SAME")
        # shape (batch_size,attn_length,1,attention_vec_size)

        self.linear_layer =  tf.keras.layers.Dense(self.vector_size)
            # shape (batch_size, attention_vec_size)

        self.coverage_layer =  tf.keras.layers.Conv2D(self.attention_length,(1,self.vector_size), padding= "SAME")
        # c has shape (batch_size, attn_length, 1, attention_vec_size)

        super(AttentionLayer, self).build(unused_input_shapes)

    def __call__(self,
                 encoder_state,
                 decoder_state,
                 input_mask,
                 coverage=None,
                 **kwargs):
        inputs = tf_utils.pack_inputs([encoder_state,decoder_state, input_mask,coverage ])
        return super(AttentionLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        unpacked_inputs = tf_utils.unpack_inputs(inputs)

        encoder_states = unpacked_inputs[0]
        input_shape = tf_utils.get_shape_list(encoder_states, expected_rank=[2, 3])
        batch_size = input_shape[0]

        decoder_states=unpacked_inputs[1]
        input_mask=unpacked_inputs[2]
        coverage  =unpacked_inputs[3]



        encoder_states = tf.expand_dims(encoder_states, axis=2)
        # now is shape (batch_size, attn_len, 1, attn_size)

        encoder_features = self.encoder_layer(encoder_states)
        # shape (batch_size,attn_length,1,attention_vec_size)

        decoder_features= self.linear_layer(decoder_states)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),1)
        # reshape to (batch_size, 1, 1, attention_vec_size)

        if self.use_coverage and coverage is not None:
            coverage_features=self.coverage_layer(coverage)
            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
            e = tf.reduce_sum(self.v * tf.math.tanh(encoder_features + decoder_features + coverage_features),
                                    [2, 3])  # shape (batch_size,attn_length)

        def masked_attention(e):
            """Take softmax of e then apply enc_padding_mask and re-normalize"""
            attn_dist = tf.nn.softmax(e)  # take softmax. shape (batch_size, attn_length)
            attn_dist *= input_mask  # apply mask
            masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
            return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

        if self.use_coverage and coverage is not None:  # non-first step of coverage
            # Multiply coverage vector by w_c to get coverage_features.
            coverage_features = self.coverage_layer(coverage )  # c has shape (batch_size, attn_length, 1, attention_vec_size)

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
        context_vector = tf.reduce_sum(tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                                             [1, 2])  # shape (batch_size, attn_size).
        context_vector = tf.reshape(context_vector, [-1, self.vector_size])

        return context_vector, attn_dist, coverage



class LinearLayer(tf.keras.layers.Layer):
  def __init__(self,
               output_size,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               **kwargs):
    super(LinearLayer, self).__init__(**kwargs)
    self.output_size = output_size
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Linear` layer with "
                      "non-floating point (and non-complex) "
                      "dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to "
                       "`Linear` should be defined. "
                       "Found `None`.")
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
    self.kernel = self.add_weight(
        "kernel",
        shape=[last_dim, self.output_size],
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
    """Implements call() for LinearLayer.

    Args:
      inputs: float Tensor of shape [batch, , ].

    Returns:
      A 3D Tensor.
    """
    ret = tf.einsum("abc,cd->abd", inputs, self.kernel)
    ret += self.bias
    if self.activation is not None:
      return self.activation(ret)
    return ret

