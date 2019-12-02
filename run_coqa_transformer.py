
"""Run BERT on CoQA, adapted from Google's run_Squad.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

# pylint: disable=unused-import,g-import-not-at-top,redefined-outer-name,reimported
from modeling import model_training_utils
from bert import bert_modeling as bert_modeling
from models import coqa_models
from bert import optimization
from bert import common_flags
from bert import input_pipeline
from bert import model_saving_utils
import coqa_end2end_lib
from bert import tokenization
from utils.misc import keras_utils
from utils.misc import tpu_lib

flags.DEFINE_enum(
    'mode', 'predict',
    ['train_and_predict', 'train', 'predict', 'export_only'],
    'One of {"train_and_predict", "train", "predict", "export_only"}. '
    '`train_and_predict`: both train and predict to a json file. '
    '`train`: only trains the model. '
    '`predict`: predict answers from the coqa json file. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`.')
flags.DEFINE_string('train_data_path', '',
                    'Training data path with train tfrecords.')
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
# Model training specific flags.
flags.DEFINE_integer('train_batch_size', 1, 'Total batch size for training.')
# Predict processing related.
flags.DEFINE_string('predict_file', None,
                    'Prediction data path with train tfrecords.')
flags.DEFINE_string('vocab_file', None,
                    'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_bool(
    'do_lower_case', True,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool(
    'verbose_logging', False,
    'If true, all of the warnings related to data processing will be printed. '
    'A number of warnings are expected for a normal SQuAD evaluation.')
flags.DEFINE_integer('predict_batch_size', 1,
                     'Total batch size for prediction.')
flags.DEFINE_integer(
    'n_best_size', 20,
    'The total number of n-best predictions to generate in the '
    'nbest_predictions.json output file.')
flags.DEFINE_integer(
    'max_answer_length', 30,
    'The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another.')
flags.DEFINE_float('cov_loss_wt', 1.0,
                   'Weight of coverage loss (lambda in the paper). '
                   'If zero, then no incentive to minimize coverage loss.')
flags.DEFINE_boolean('use_pointer_gen', True,
                     'If True, use pointer-generator model. If False, use baseline model.')

flags.DEFINE_integer(
    'max_oov_size', 10,
    'The maximum number of possible OOV words per input sequence.  ')

flags.DEFINE_integer('batch_size', 4, 'minibatch size')
flags.DEFINE_integer('beam_size', 1, 'beam size')


common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def coqa_loss_fn( final_dists,
                  target_words_ids,
                  dec_padding_mask,
                  loss_factor=1.0):

    # Calculate the loss per step
    # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
    loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
    #
    # batch_nums = tf.range(0, limit=FLAGS.batch_size) # shape (batch_size)
    # for dec_step, dist in enumerate(final_dists):
    #     targets = target_words_ids[:,dec_step] # The indices of the target words. shape (batch_size)
    #     indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
    #     gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
    #     losses = -tf.math.log(gold_probs)
    #     loss_per_step.append(losses)

    depth = final_dists.shape.as_list()[-1]
    dec_target_ohe = tf.one_hot(target_words_ids, depth=depth)
    losses=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=final_dists, labels=dec_target_ohe)

    #losses= tf.nn.sparse_softmax_cross_entropy_with_logits(target_words_ids,final_dists)
    # Apply dec_padding_mask and get loss
    loss_per_step= tf.unstack(losses,axis=1)
    _loss = _mask_and_avg(loss_per_step, dec_padding_mask)

    _total_loss = _loss

    return  _total_loss

def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """
  padding_mask=tf.cast(padding_mask,tf.dtypes.float32)
  dec_lens = (tf.reduce_sum(padding_mask, axis=1)) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss

def get_loss_fn(loss_factor=1.0):
  """Gets a loss function for coqa task."""

  def _loss_fn(labels, model_outputs):
    target_words_ids = labels['answer_ids']
    target_words_mask = labels['answer_mask']
    #unique_ids,final_dists, attn_dists = model_outputs
    unique_ids, final_dists  = model_outputs
    return coqa_loss_fn(final_dists,
                        #attn_dists,
                        target_words_ids,
                        target_words_mask,
                        loss_factor=loss_factor)

  return _loss_fn


def get_raw_results(predictions):
  """Converts multi-replica predictions to RawResult."""
  for unique_id, sentence_ids  in zip(predictions['unique_ids'],
                                                  predictions['sentence_ids'] ):
      yield coqa_end2end_lib.RawResultEnd2end(
          unique_id=unique_id,
          sentence_ids=sentence_ids)


def predict_coqa_customized(strategy, input_meta_data, bert_config,
                             predict_tfrecord_path, num_steps):
  """Make predictions using a Bert-based coqa model."""
  primary_cpu_task = '/job:worker' if FLAGS.tpu else ''

  num_train_examples = input_meta_data['train_data_size']
  max_seq_length = input_meta_data['max_seq_length']
  max_answer_length = input_meta_data['max_answer_length']

  # add use_pointer_gen
  bert_config.add_from_dict({"use_pointer_gen": FLAGS.use_pointer_gen})
  # max_oov_size  let's just add something for now
  bert_config.add_from_dict({"max_oov_size": FLAGS.max_oov_size})
  bert_config.add_from_dict({"max_seq_length": max_seq_length})
  bert_config.add_from_dict({"max_answer_length": max_answer_length})

  with tf.device(primary_cpu_task):
    predict_dataset = input_pipeline.create_coqa_dataset_seq2seq(
        predict_tfrecord_path,
        input_meta_data['max_seq_length'],
        max_answer_length,
        FLAGS.predict_batch_size,
        is_training=False)
    predict_iterator = iter(
        strategy.experimental_distribute_dataset(predict_dataset))

    with strategy.scope():
      # Prediction always uses float32, even if training uses mixed precision.
      #tf.keras.mixed_precision.experimental.set_policy('float32')
      coqa_model, _ = coqa_models.coqa_model_transformer(
          config=bert_config,
          max_seq_length=input_meta_data['max_seq_length'],
          max_answer_length=max_answer_length,
          float_type=tf.float32)

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    logging.info('Restoring checkpoints from %s', checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=coqa_model)
    checkpoint.restore(checkpoint_path).expect_partial()

    def decode_sequence(x):

        pred_ids=x['decode_ids'].numpy()
        pred_mask =x['decode_mask'].numpy()



        for i in range(1, bert_config.max_answer_length):
            unique_ids, logits = coqa_model(
                                inputs = ( {
                                    'unique_ids' : x['unique_ids'],
                                    'input_word_ids' : x['input_word_ids'],
                                    'input_type_ids' : x['input_type_ids'],
                                    'input_mask':x['input_mask'],
                                    'decode_ids':tf.convert_to_tensor(pred_ids),
                                    'decode_mask':tf.convert_to_tensor(pred_mask)
                                }),
                                training=False)

            next_pred = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()

            # Only update the i-th column in one step.
            pred_ids[:, i] = next_pred[:, i - 1]
            pred_mask[:, i] = tf.cast(tf.not_equal(next_pred[:, i - 1], 105),tf.int32) #tf.not_equal(next_pred[:, i - 1], 105)
            #pred_mask[:,i]
        return x['unique_ids'], pred_ids


    @tf.function
    def predict_step(iterator):
      """Predicts on distributed devices."""

      def _replicated_step(inputs):
        """Replicated prediction calculation."""
        x, _ = inputs
        unique_ids, sentence_ids  = decode_sequence(x)

        return dict(
            unique_ids=unique_ids,
            sentence_ids=sentence_ids )

      outputs = strategy.experimental_run_v2(
          _replicated_step, args=(next(iterator),))
      return outputs

    all_results = []
    for _ in range(num_steps):
      predictions = predict_step(predict_iterator)

      for result in get_raw_results(predictions):
        all_results.append(result)
      if len(all_results) % 100 == 0:
        logging.info('Made predictions for %d records.', len(all_results))
    return all_results


def train_coqa(strategy,
                input_meta_data,
                custom_callbacks=None,
                run_eagerly=False):
  """Run bert coqa training."""
  if strategy:
    logging.info('Training using customized training loop with distribution'
                 ' strategy.')
  # Enables XLA in Session Config. Should not be set for TPU.
  keras_utils.set_config_v2(FLAGS.enable_xla)

  use_float16 = common_flags.use_float16()
  if use_float16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  # add some extra to bert_config
  bert_config.add_from_dict(input_meta_data)

  num_train_examples = input_meta_data['train_data_size']
  max_seq_length = input_meta_data['max_seq_length']
  max_answer_length = input_meta_data['max_answer_length']

  # add use_pointer_gen
  bert_config.add_from_dict({"use_pointer_gen":FLAGS.use_pointer_gen})
  #max_oov_size  let's just add something for now
  bert_config.add_from_dict({"max_oov_size": FLAGS.max_oov_size})
  bert_config.add_from_dict({"max_seq_length":max_seq_length})
  bert_config.add_from_dict({"max_answer_length": max_answer_length})


  epochs = FLAGS.num_train_epochs

  steps_per_epoch = int(num_train_examples / FLAGS.train_batch_size)
  warmup_steps = int(epochs * num_train_examples * 0.1 / FLAGS.train_batch_size)
  train_input_fn = functools.partial(
      input_pipeline.create_coqa_dataset_seq2seq,
      FLAGS.train_data_path,
      max_seq_length,
      max_answer_length,
      FLAGS.train_batch_size,
      is_training=True)

  def _get_coqa_model():
    """Get Squad model and optimizer."""
    coqa_model, core_model  = coqa_models.coqa_model_transformer(
        bert_config,
        max_seq_length,
        max_answer_length,
        float_type=tf.float16 if use_float16 else tf.float32)
    coqa_model.optimizer = optimization.create_optimizer(
        FLAGS.learning_rate, steps_per_epoch * epochs, warmup_steps)
    if use_float16:
      # Wraps optimizer with a LossScaleOptimizer. This is done automatically
      # in compile() with the "mixed_float16" policy, but since we do not call
      # compile(), we must wrap the optimizer manually.
      coqa_model.optimizer = (
          tf.keras.mixed_precision.experimental.LossScaleOptimizer(
              coqa_model.optimizer, loss_scale=common_flags.get_loss_scale()))
    if FLAGS.fp16_implementation == 'graph_rewrite':
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      coqa_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          coqa_model.optimizer)
    return coqa_model,core_model

  # The original BERT model does not scale the loss by
  # 1/num_replicas_in_sync. It could be an accident. So, in order to use
  # the same hyper parameter, we do the same thing here by keeping each
  # replica loss as it is.
  loss_fn = get_loss_fn(
      loss_factor=1.0 /
      strategy.num_replicas_in_sync if FLAGS.scale_loss else 1.0)

  model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_coqa_model,
      loss_fn=loss_fn,
      model_dir=FLAGS.model_dir,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=FLAGS.steps_per_loop,
      epochs=epochs,
      train_input_fn=train_input_fn,
      init_checkpoint= None,#FLAGS.init_checkpoint, here we don't use BERT
      run_eagerly=run_eagerly,
      custom_callbacks=custom_callbacks)


def predict_coqa(strategy, input_meta_data):
  """Makes predictions for a coqa dataset."""
  bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  doc_stride = input_meta_data['doc_stride']
  max_seq_length=input_meta_data['max_seq_length']
  max_query_length = input_meta_data['max_query_length']
  max_answer_length=  input_meta_data['max_answer_length']
  # Whether data should be in Ver 2.0 format.

  eval_examples = coqa_end2end_lib.read_coqa_examples(
      input_file=FLAGS.predict_file,
      is_training=False )

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  eval_writer = coqa_end2end_lib.FeatureWriter(
      filename=os.path.join(FLAGS.model_dir, 'eval.tf_record'),
      is_training=False)
  eval_features = []

  def _append_feature(feature, is_padding = False):
    if not is_padding:
      eval_features.append(feature)
    eval_writer.process_feature(feature)

  # TPU requires a fixed batch size for all batches, therefore the number
  # of examples must be a multiple of the batch size, or else examples
  # will get dropped. So we pad with fake examples which are ignored
  # later on.
  dataset_size = coqa_end2end_lib.convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      max_answer_length=max_answer_length,
      is_training=False,
      output_fn=_append_feature  )
  eval_writer.close()

  logging.info('***** Running predictions *****')
  logging.info('  Num orig examples = %d', len(eval_examples))
  logging.info('  Num split examples = %d', len(eval_features))
  logging.info('  Batch size = %d', FLAGS.predict_batch_size)

  num_steps = int(dataset_size / FLAGS.predict_batch_size)
  all_results = predict_coqa_customized(strategy, input_meta_data, bert_config,
                                         eval_writer.filename, num_steps)

  output_prediction_file = os.path.join(FLAGS.model_dir, 'predictions.json')
  output_null_log_odds_file = os.path.join(FLAGS.model_dir, 'null_odds.json')
  StartToken='[START]'
  StopToken='[STOP]'

  coqa_end2end_lib.write_predictions_end2end(
      eval_examples,
      eval_features,
      all_results,
      output_prediction_file,
      FLAGS.max_answer_length,
      tokenizer,
      StartToken,
      StopToken
       )


def export_coqa(model_export_path, input_meta_data):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  if not model_export_path:
    raise ValueError('Export path is not specified: %s' % model_export_path)
  bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  coqa_model = coqa_models.coqa_model_transformer(
      bert_config,
      input_meta_data['max_seq_length'],
      float_type=tf.float32)
  model_saving_utils.export_bert_model(
      model_export_path, model=coqa_model, checkpoint_dir=FLAGS.model_dir)


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  #tf.enable_eager_execution()
  #tf.compat.v1.enable_eager_execution()

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))


  if FLAGS.mode == 'export_only':
    export_coqa(FLAGS.model_export_path, input_meta_data)
    return

  strategy = None
  if FLAGS.strategy_type == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
  elif FLAGS.strategy_type == 'multi_worker_mirror':
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  elif FLAGS.strategy_type == 'tpu':
    cluster_resolver = tpu_lib.tpu_initialize(FLAGS.tpu)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
  else:
    raise ValueError('The distribution strategy type is not supported: %s' %
                     FLAGS.strategy_type)
  if FLAGS.mode in ('train', 'train_and_predict'):
    train_coqa(strategy, input_meta_data)
  if FLAGS.mode in ('predict', 'train_and_predict'):
    predict_coqa(strategy, input_meta_data)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('model_dir')

  tf.compat.v1.reset_default_graph()

  app.run(main)


