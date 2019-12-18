# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BERT finetuning task dataset generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app
from absl import flags
import tensorflow as tf

from bert import classifier_data_lib
import squad_lib
import coqa_lib
import coqa_end2end_lib
import coqa_bert_lib
import coqa_bert_span_lib
import coqa_span_rationale_tag_lib
import coqa_bert_rt_transformer_lib

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "fine_tuning_task_type", "classification", ["bert_rt_transformer", "coqa_span_rationale_tag","classification", "squad","coqa","coqa_end2end","coqa_bert_e2e","coqa_bert_span"],
    "The name of the BERT fine tuning task for which data "
    "will be generated..")

# BERT classification specific flags.
flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_enum("classification_task_name", "MNLI",
                  ["COLA", "MNLI", "MRPC", "XNLI"],
                  "The name of the task to train BERT classifier.")

# BERT Squad task specific flags.
flags.DEFINE_string(
    "data_file", None,
    "The input data file in for generating training data for BERT squad task.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 128,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    'max_answer_length', 30,
    'The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another.')

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

# Shared flags across BERT fine-tuning tasks.
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "train_data_output_path", None,
    "The path in which generated training input data will be written as tf"
    " records."
)

flags.DEFINE_string(
    "eval_data_output_path", None,
    "The path in which generated training input data will be written as tf"
    " records."
)

flags.DEFINE_string("meta_data_file_path", None,
                    "The path in which input meta data will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


def generate_classifier_dataset():
  """Generates classifier dataset and returns input meta data."""
  assert FLAGS.input_data_dir and FLAGS.classification_task_name

  processors = {
      "cola": classifier_data_lib.ColaProcessor,
      "mnli": classifier_data_lib.MnliProcessor,
      "mrpc": classifier_data_lib.MrpcProcessor,
      "xnli": classifier_data_lib.XnliProcessor,
  }
  task_name = FLAGS.classification_task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()
  return classifier_data_lib.generate_tf_record_from_data_file(
      processor,
      FLAGS.input_data_dir,
      FLAGS.vocab_file,
      train_data_output_path=FLAGS.train_data_output_path,
      eval_data_output_path=FLAGS.eval_data_output_path,
      max_seq_length=FLAGS.max_seq_length,
      do_lower_case=FLAGS.do_lower_case)


def generate_squad_dataset():
  """Generates squad training dataset and returns input meta data."""
  assert FLAGS.data_file
  return squad_lib.generate_tf_record_from_json_file(
      FLAGS.data_file, FLAGS.vocab_file, FLAGS.train_data_output_path,
      FLAGS.max_seq_length, FLAGS.do_lower_case, FLAGS.max_query_length,
      FLAGS.doc_stride, FLAGS.version_2_with_negative)
def generate_coqa_dataset():
  """Generates squad training dataset and returns input meta data."""
  assert FLAGS.data_file
  return coqa_lib.generate_tf_record_from_json_file(
      FLAGS.data_file, FLAGS.vocab_file, FLAGS.train_data_output_path,
      FLAGS.max_seq_length, FLAGS.do_lower_case, FLAGS.max_query_length,
      FLAGS.doc_stride )

def generate_coqa_end2end_dataset():
  """Generates squad training dataset and returns input meta data."""
  assert FLAGS.data_file
  return coqa_end2end_lib.generate_tf_record_from_json_file(
      FLAGS.data_file, FLAGS.vocab_file, FLAGS.train_data_output_path,
      FLAGS.max_seq_length, FLAGS.max_answer_length, FLAGS.do_lower_case, FLAGS.max_query_length,
      FLAGS.doc_stride )

def generate_coqa_bert_e2e_dataset():
    assert FLAGS.data_file
    return coqa_bert_lib.generate_tf_record_from_json_file(
        FLAGS.data_file, FLAGS.vocab_file, FLAGS.train_data_output_path,
        FLAGS.max_seq_length, FLAGS.max_answer_length, FLAGS.do_lower_case, FLAGS.max_query_length,
        FLAGS.doc_stride)
def generate_coqa_bert_span_dataset():
    assert FLAGS.data_file
    return coqa_bert_span_lib.generate_tf_record_from_json_file(
        FLAGS.data_file, FLAGS.vocab_file, FLAGS.train_data_output_path,
        FLAGS.max_seq_length, FLAGS.max_answer_length, FLAGS.do_lower_case, FLAGS.max_query_length,
        FLAGS.doc_stride)

def generate_coqa_span_rationale_tag_dataset():
    assert FLAGS.data_file
    return coqa_span_rationale_tag_lib.generate_tf_record_from_json_file(
        FLAGS.data_file, FLAGS.vocab_file, FLAGS.train_data_output_path,
        FLAGS.max_seq_length, FLAGS.max_answer_length, FLAGS.do_lower_case, FLAGS.max_query_length,
        FLAGS.doc_stride)
def generate_coqa_bert_rt_transformer_dataset():
    assert FLAGS.data_file
    return coqa_bert_rt_transformer_lib.generate_tf_record_from_json_file(
        FLAGS.data_file, FLAGS.vocab_file, FLAGS.train_data_output_path,
        FLAGS.max_seq_length, FLAGS.max_answer_length, FLAGS.do_lower_case, FLAGS.max_query_length,
        FLAGS.doc_stride)

def main(_):
  if FLAGS.fine_tuning_task_type == "classification":
    input_meta_data = generate_classifier_dataset()
  elif FLAGS.fine_tuning_task_type == "squad":
    input_meta_data = generate_squad_dataset()
  elif FLAGS.fine_tuning_task_type == "coqa":
    input_meta_data=generate_coqa_dataset()
  elif FLAGS.fine_tuning_task_type == "coqa_end2end":
      input_meta_data=generate_coqa_end2end_dataset()
  elif FLAGS.fine_tuning_task_type == "coqa_bert_e2e":
      input_meta_data=generate_coqa_bert_e2e_dataset()
  elif FLAGS.fine_tuning_task_type == "coqa_bert_span":
      input_meta_data = generate_coqa_bert_span_dataset()
  elif FLAGS.fine_tuning_task_type =="coqa_span_rationale_tag":
      input_meta_data = generate_coqa_span_rationale_tag_dataset()
  elif FLAGS.fine_tuning_task_type=="bert_rt_transformer":
      input_meta_data = generate_coqa_bert_rt_transformer_dataset()

  with tf.io.gfile.GFile(FLAGS.meta_data_file_path, "w") as writer:
    writer.write(json.dumps(input_meta_data, indent=4) + "\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("train_data_output_path")
  flags.mark_flag_as_required("meta_data_file_path")
  app.run(main)
