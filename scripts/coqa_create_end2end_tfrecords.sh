#!/usr/bin/env bash


export COQA_DIR=../datasets/CoQA
export BERT_BASE_DIR=../pretrained/uncased_L-12_H-768_A-12

python ../create_finetuning_data.py \
 --data_file=${COQA_DIR}/coqa-train-v1.0.sample.json \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --train_data_output_path=${COQA_DIR}/coqa_e2e_train.tf_record \
 --meta_data_file_path=${COQA_DIR}/coqa_e2e_meta_data \
 --fine_tuning_task_type=coqa-pgnet --max_seq_length=384


