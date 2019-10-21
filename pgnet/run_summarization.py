
 #"""Run  training for PGNet in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

# pylint: disable=unused-import,g-import-not-at-top,redefined-outer-name,reimported
from modeling import model_training_utils
from bert import optimization
from bert import common_flags
from bert import input_pipeline
from bert import model_saving_utils
from utils.misc import tpu_lib

flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')
# Model training specific flags.
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('num_steps_per_epoch', 1000,
                     'Total number of training steps to run per epoch.')
flags.DEFINE_float('warmup_steps', 10000,
                   'Warmup steps for Adam weight decay optimizer.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def get_pretrain_input_data(input_file_pattern, seq_length,
                            max_predictions_per_seq, batch_size, strategy):
  """Returns input dataset from input file string."""

  # When using TPU pods, we need to clone dataset across
  # workers and need to pass in function that returns the dataset rather
  # than passing dataset instance itself.
  use_dataset_fn = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
  if use_dataset_fn:
    if batch_size % strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    batch_size = int(batch_size / strategy.num_replicas_in_sync)

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    input_files = []
    for input_pattern in input_file_pattern.split(','):
      input_files.extend(tf.io.gfile.glob(input_pattern))

    train_dataset = input_pipeline.create_pretrain_dataset(
        input_files,
        seq_length,
        max_predictions_per_seq,
        batch_size,
        is_training=True,
        input_pipeline_context=ctx)
    return train_dataset

  return _dataset_fn if use_dataset_fn else _dataset_fn()


def get_loss_fn(loss_factor=1.0):
  """Returns loss function for BERT pretraining."""

  def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
    return tf.keras.backend.mean(losses) * loss_factor

  return _bert_pretrain_loss_fn


def run_customized_training(strategy,
                            bert_config,
                            max_seq_length,
                            max_predictions_per_seq,
                            model_dir,
                            steps_per_epoch,
                            steps_per_loop,
                            epochs,
                            initial_lr,
                            warmup_steps,
                            input_files,
                            train_batch_size):
  """Run BERT pretrain model training using low-level API."""

  train_input_fn = functools.partial(get_pretrain_input_data, input_files,
                                     max_seq_length, max_predictions_per_seq,
                                     train_batch_size, strategy)

  def _get_pretrain_model():
    """Gets a pretraining model."""
    pretrain_model, core_model = bert_models.pretrain_model(
        bert_config, max_seq_length, max_predictions_per_seq)
    pretrain_model.optimizer = optimization.create_optimizer(
        initial_lr, steps_per_epoch * epochs, warmup_steps)
    if FLAGS.fp16_implementation == 'graph_rewrite':
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      pretrain_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          pretrain_model.optimizer)
    return pretrain_model, core_model

  trained_model = model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_pretrain_model,
      loss_fn=get_loss_fn(
          loss_factor=1.0 /
          strategy.num_replicas_in_sync if FLAGS.scale_loss else 1.0),
      model_dir=model_dir,
      train_input_fn=train_input_fn,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=steps_per_loop,
      epochs=epochs)

  # Creates the BERT core model outside distribution strategy scope.
  _, core_model = bert_models.pretrain_model(bert_config, max_seq_length,
                                             max_predictions_per_seq)

  # Restores the core model from model checkpoints and get a new checkpoint only
  # contains the core model.
  model_saving_utils.export_pretraining_checkpoint(
      checkpoint_dir=model_dir, model=core_model)
  return trained_model


def run_bert_pretrain(strategy):
  """Runs BERT pre-training."""

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  if not strategy:
    raise ValueError('Distribution strategy is not specified.')

  # Runs customized training loop.
  logging.info('Training using customized training loop TF 2.0 with distrubuted'
               'strategy.')

  return run_customized_training(
      strategy,
      bert_config,
      FLAGS.max_seq_length,
      FLAGS.max_predictions_per_seq,
      FLAGS.model_dir,
      FLAGS.num_steps_per_epoch,
      FLAGS.steps_per_loop,
      FLAGS.num_train_epochs,
      FLAGS.learning_rate,
      FLAGS.warmup_steps,
      FLAGS.input_files,
      FLAGS.train_batch_size)


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'
  strategy = None
  if FLAGS.strategy_type == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
  elif FLAGS.strategy_type == 'tpu':
    cluster_resolver = tpu_lib.tpu_initialize(FLAGS.tpu)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
  else:
    raise ValueError('The distribution strategy type is not supported: %s' %
                     FLAGS.strategy_type)
  if strategy:
    print('***** Number of cores used : ', strategy.num_replicas_in_sync)

  run_bert_pretrain(strategy)


if __name__ == '__main__':
  app.run(main)