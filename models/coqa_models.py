

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import tensorflow_hub as hub

from modeling import tf_utils
import bert.bert_modeling  as bert_modeling
import bert.bert_models as bert_models
import models.coqa_layers as coqalayers

def coqa_model_bert_transformer(config, max_seq_length, max_answer_length, float_type, training=False,
                      initializer=None):
    unique_ids = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name='unique_ids')
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')
    decode_ids = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='decode_ids')
    decode_mask = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='decode_mask')


    bert_model = bert_modeling.get_bert_model(
        input_word_ids,
        input_mask,
        input_type_ids,
        config= config,
        name='bert_model',
        float_type=float_type)

    # `Bert Coqa Pgnet Model` only uses the sequence_output which
    # has dimensionality (batch_size, sequence_length, num_hidden).
    sequence_output = bert_model.outputs[1]

    if initializer is None:
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=config.initializer_range)

    #
    coqa_layer = coqalayers.SimpleTransformerDecoder(config=config,
                                              name='simple_transformer_decoder')

    final_dists = coqa_layer(sequence_output,
                             input_mask,
                             decode_ids,
                             decode_mask
                             )

    coqa = tf.keras.Model(
        inputs=({
                    'unique_ids': unique_ids,
                    'input_word_ids': input_word_ids,
                    'input_type_ids': input_type_ids,
                    'input_mask': input_mask,
                    'decode_ids': decode_ids,
                    'decode_mask': decode_mask},),

        outputs=[unique_ids, final_dists])


    return coqa, bert_model


def coqa_model_transformer(config, max_seq_length, max_answer_length, float_type, training=False,
                      initializer=None):
    unique_ids = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name='unique_ids')
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')
    decode_ids = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='decode_ids')
    decode_mask = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='decode_mask')

    bert_model = tf.keras.Model()

    # bert_model = bert_modeling.get_bert_model(
    #     input_word_ids,
    #     input_mask,
    #     input_type_ids,
    #     config= config,
    #     name='bert_model',
    #     float_type=float_type)
    #
    # # `Bert Coqa Pgnet Model` only uses the sequence_output which
    # # has dimensionality (batch_size, sequence_length, num_hidden).
    # sequence_output = bert_model.outputs[1]
    #
    # if initializer is None:
    #     initializer = tf.keras.initializers.TruncatedNormal(
    #         stddev=config.initializer_range)
    #
    #
    coqa_layer = coqalayers.SimpleTransformer(config=config,
                                              name='simple_transformer')

    final_dists = coqa_layer(input_word_ids,
                             input_mask,
                             input_type_ids,
                             decode_ids,
                             decode_mask
                             )

    coqa = tf.keras.Model(
        inputs=({
                    'unique_ids': unique_ids,
                    'input_word_ids': input_word_ids,
                    'input_type_ids': input_type_ids,
                    'input_mask': input_mask,
                    'decode_ids': decode_ids,
                    'decode_mask': decode_mask},),

        outputs=[unique_ids, final_dists])


    return coqa, bert_model


def coqa_model_transformer2heads(config, max_seq_length, max_answer_length, float_type, training=False,
                      initializer=None):
    unique_ids = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name='unique_ids')
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')
    decode_ids = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='decode_ids')
    decode_mask = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='decode_mask')

    bert_model = tf.keras.Model()

    # bert_model = bert_modeling.get_bert_model(
    #     input_word_ids,
    #     input_mask,
    #     input_type_ids,
    #     config= config,
    #     name='bert_model',
    #     float_type=float_type)
    #
    # # `Bert Coqa Pgnet Model` only uses the sequence_output which
    # # has dimensionality (batch_size, sequence_length, num_hidden).
    # sequence_output = bert_model.outputs[1]
    #
    # if initializer is None:
    #     initializer = tf.keras.initializers.TruncatedNormal(
    #         stddev=config.initializer_range)
    #
    #
    coqa_layer = coqalayers.SimpleTransformer2Heads(config=config,
                                              name='simple_transformer')

    final_dists, sequence_output = coqa_layer(input_word_ids,
                             input_mask,
                             input_type_ids,
                             decode_ids,
                             decode_mask
                             )
    span_logits_layer = BertSpanLogitsLayer(
        initializer=initializer, float_type=float_type, name='squad_logits')
    start_logits, end_logits = span_logits_layer(sequence_output)

    coqa = tf.keras.Model(
        inputs=({
                    'unique_ids': unique_ids,
                    'input_word_ids': input_word_ids,
                    'input_type_ids': input_type_ids,
                    'input_mask': input_mask,
                    'decode_ids': decode_ids,
                    'decode_mask': decode_mask},),

        outputs=[unique_ids, final_dists, start_logits, end_logits ])


    return coqa, bert_model


def coqa_modelseq2seq(config, max_seq_length, max_answer_length, max_oov_size, float_type, training=False,
               initializer=None):
    
    unique_ids = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name='unique_ids')
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')
    decode_ids = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='decode_ids')

    bert_model = tf.keras.Model()

    # bert_model = bert_modeling.get_bert_model(
    #     input_word_ids,
    #     input_mask,
    #     input_type_ids,
    #     config= config,
    #     name='bert_model',
    #     float_type=float_type)
    #
    # # `Bert Coqa Pgnet Model` only uses the sequence_output which
    # # has dimensionality (batch_size, sequence_length, num_hidden).
    # sequence_output = bert_model.outputs[1]
    #
    # if initializer is None:
    #     initializer = tf.keras.initializers.TruncatedNormal(
    #         stddev=config.initializer_range)
    #
    # #
    # # Double headed- trained on both span positions and final answer
    #
    # coqa_logits_layer = bert_models.BertCoqaLogitsLayer(
    #     initializer=initializer, float_type=float_type, name='coqa_logits')
    # start_logits, end_logits = coqa_logits_layer(sequence_output)
    #
    # #figure out the span text from the start logits and end_logits here.
    # span_text_ids,span_mask=get_best_span_prediction(input_word_ids, start_logits, end_logits,max_seq_length )

    # pgnet_model_layer =modeling.PGNetSummaryModel(config=config ,
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

    # PGNet only: end to end - question+context to answer

    coqa_layer = coqalayers.SimpleLSTMSeq2Seq(config=config,
                                              training=training,
                                              name = 'simple_lstm_seq2seq')

    final_dists = coqa_layer(input_word_ids,
                             input_mask,
                             input_type_ids,
                             decode_ids,
                             )

    coqa = tf.keras.Model(
        inputs=({
                    'unique_ids': unique_ids,
                    'input_word_ids': input_word_ids,
                    'input_type_ids':input_type_ids,
                    'input_mask': input_mask,
                    'decode_ids': decode_ids},),

        outputs=[unique_ids, final_dists])


    # Bert+PGNet:  end to end
    # pgnet_model_layer = modeling.PGNetDecoderModel(config=config ,
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

    return coqa, bert_model


def coqa_model_2heads(config, max_seq_length, max_answer_length, float_type, training=False,
                      initializer=None):
    unique_ids = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name='unique_ids')
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')
    decode_ids = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='decode_ids')

    bert_model = bert_modeling.get_bert_model(
        input_word_ids,
        input_mask,
        input_type_ids,
        config=config,
        name='bert_model',
        float_type=float_type)

    # `Bert Coqa Pgnet Model` only uses the sequence_output which
    # has dimensionality (batch_size, sequence_length, num_hidden).
    sequence_output = bert_model.outputs[1]

    if initializer is None:
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=config.initializer_range)

    #
    # Double headed- trained on both span positions and final answer

    coqa_logits_layer = bert_models.BertCoqaLogitsLayer(
        initializer=initializer, float_type=float_type, name='coqa_logits')
    start_logits, end_logits = coqa_logits_layer(sequence_output)

    # figure out the span text from the start logits and end_logits here.
    span_text_ids, span_mask = get_best_span_prediction(input_word_ids, start_logits, end_logits, max_seq_length)

    # pgnet_model_layer =modeling.PGNetSummaryModel(config=config ,
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

    # PGNet only: end to end - question+context to answer

    coqa_layer = coqalayers.SimpleLSTMSeq2Seq(config=config,
                                              training=training,
                                              name='simple_lstm_seq2seq')

    final_dists = coqa_layer(span_text_ids,
                             span_mask,
                             decode_ids,
                             )

    coqa = tf.keras.Model(
        inputs=({
                    'unique_ids': unique_ids,
                    'input_word_ids': input_word_ids,
                    'input_type_ids': input_type_ids,
                    'input_mask': input_mask,
                    'decode_ids': decode_ids},),

        outputs=[unique_ids, final_dists, start_logits, end_logits])

    # Bert+PGNet:  end to end
    # pgnet_model_layer = modeling.PGNetDecoderModel(config=config ,
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

    return coqa, bert_model
def transformer_encoder_decoder(model,config):
    return model.get_encoder(),model.get_decoder()

def one_step_decoder_model(model,config):

    ######### encoder for one step decoder
    input_word_ids = model.input[0]['input_word_ids'] #input_word_ids
    input_mask = model.input[0]['input_mask']  # input_word_ids
    input_type_ids = model.input[0]['input_type_ids']  # input_word_ids


    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('simple_lstm_seq2seq')._layers[2].output

    enc_outputs = tf.expand_dims(encoder_outputs, 2)

    enc_features = model.get_layer('simple_lstm_seq2seq')._layers[3](enc_outputs)


    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.Model([input_word_ids,input_mask,input_type_ids], [encoder_states,enc_features])

    ########################
    #one step decoder
    decoder_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='one_decoder_input')
    decoder_state_input_h = tf.keras.layers.Input(shape=(config.hidden_size,), name='state_h_input')
    decoder_state_input_c = tf.keras.layers.Input(shape=(config.hidden_size,), name='state_c_input')
    enc_input_feature = tf.keras.layers.Input(shape=(config.max_seq_length,1,config.hidden_size,), name='encoder_input_feature')
    enc_input_mask= tf.keras.layers.Input( shape=(config.max_seq_length,), dtype=tf.int32, name='enc_input_mask')

    decoder_input_feature = [ model.get_layer('simple_lstm_seq2seq')._layers[0](x) for x in tf.unstack(decoder_input,axis=1)]
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.get_layer('simple_lstm_seq2seq')._layers[6]
    #model._layers['simple_lstm_seq2seq']._layers['decoder']

    decoder_outputs, decoder_states= decoder_lstm(
        decoder_input_feature[0],  decoder_states_inputs)

    context = model.get_layer('simple_lstm_seq2seq')._layers[4](decoder_state=decoder_states, encoder_features= enc_input_feature, input_mask= enc_input_mask)

    output = model.get_layer('simple_lstm_seq2seq')._layers[5]([[decoder_outputs], [context]])

    projector = model.get_layer('simple_lstm_seq2seq')._layers[7]

    decoded_dist = projector([output])

    _, decoded_id = tf.nn.top_k(decoded_dist, 1)

    decoded_id=tf.squeeze(decoded_id,axis=0)

    decoder_model = tf.keras.Model(
        [decoder_input] + decoder_states_inputs+[enc_input_mask]+[enc_input_feature],
        [decoded_id] + decoder_states)

    return encoder_model, decoder_model

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


class BertSpanLogitsLayer(tf.keras.layers.Layer):
  """Returns a layer that computes custom logits for BERT squad model."""

  def __init__(self, initializer=None, float_type=tf.float32, **kwargs):
    super(BertSpanLogitsLayer, self).__init__(**kwargs)
    self.initializer = initializer
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.final_dense = tf.keras.layers.Dense(
        units=2, kernel_initializer=self.initializer, name='final_dense')
    super(BertSpanLogitsLayer, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""
    sequence_output = inputs

    input_shape = sequence_output.shape.as_list()
    sequence_length = input_shape[1]
    num_hidden_units = input_shape[2]

    final_hidden_input = tf.keras.backend.reshape(sequence_output,
                                                  [-1, num_hidden_units])
    logits = self.final_dense(final_hidden_input)
    logits = tf.keras.backend.reshape(logits, [-1, sequence_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])
    unstacked_logits = tf.unstack(logits, axis=0)
    if self.float_type == tf.float16:
      unstacked_logits = tf.cast(unstacked_logits, tf.float32)
    return unstacked_logits[0], unstacked_logits[1]

