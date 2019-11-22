

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
    # Double headed- trained on both span positions and final answer

    coqa_logits_layer = bert_models.BertCoqaLogitsLayer(
        initializer=initializer, float_type=float_type, name='coqa_logits')
    start_logits, end_logits = coqa_logits_layer(sequence_output)

    #figure out the span text from the start logits and end_logits here.
    span_text_ids,span_mask=get_best_span_prediction(input_word_ids, start_logits, end_logits,max_seq_length )

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

    final_dists = coqa_layer(span_text_ids,
                             span_mask,
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

def one_step_decoder_model(model,config):

    #########
    encoder_inputs1 = model.input[0]['input_word_ids'] #input_word_ids
    encoder_inputs2 = model.input[0]['input_mask']  # input_word_ids
    encoder_inputs3 = model.input[0]['input_type_ids']  # input_word_ids



    encoder_outputs, state_h_enc, state_c_enc = model.layers[2704]._layers[1].output
    #model._layers['simple_lstm_seq2seq']._layers['encoder'].output


    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.Model([encoder_inputs1,encoder_inputs2,encoder_inputs3], encoder_states)

    ##########
    #one step decoder
    decoder_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='one_decoder_input')
    decoder_state_input_h = tf.keras.layers.Input(shape=(config.hidden_size,), name='state_h_input')
    decoder_state_input_c = tf.keras.layers.Input(shape=(config.hidden_size,), name='state_c_input')

    decoder_input_feature = [model.layers[2704]._layers[0](x) for x in tf.unstack(decoder_input,axis=1)]
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm =model.layers[2704]._layers[2]
    #model._layers['simple_lstm_seq2seq']._layers['decoder']

    decoder_outputs, decoder_states= decoder_lstm(
        decoder_input_feature[0],  decoder_states_inputs)

    projector =model.layers[2704]._layers[3]
    #model._layers['simple_lstm_seq2seq']._layers['projector']

    decoded_dist = projector([decoder_outputs])

    _, decoded_id = tf.nn.top_k(decoded_dist, 1)
    decoded_id=tf.squeeze(decoded_id,axis=0)

    decoder_model = tf.keras.Model(
        [decoder_input] + decoder_states_inputs,
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
