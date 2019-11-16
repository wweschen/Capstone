

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import tensorflow_hub as hub

from modeling import tf_utils
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
    answer_ids = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='answer_ids')
    answer_mask = tf.keras.layers.Input(
        shape=(max_answer_length,), dtype=tf.int32, name='answer_mask')

    bert_model = tf.keras.models.Model() #BERT model placeholder
     
    # `Bert Coqa Pgnet Model` only uses the sequnce_output which
    # has dimensionality (batch_size, sequence_length, num_hidden).
    # sequence_output = core_model.outputs[1]

    if initializer is None:
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=config.initializer_range)

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
                                              training=training)

    final_dists = coqa_layer(input_word_ids,
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
                    'answer_mask': answer_mask},),

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


