#!/usr/bin/env bash


export SQUAD_DIR=../datasets/squad
export SQUAD_VERSION=v1.1
export BERT_BASE_DIR=../pretrained/uncased_L-12_H-768_A-12

python ../create_finetuning_data.py \
 --data_file=${SQUAD_DIR}/train-${SQUAD_VERSION}.sample.json \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --train_data_output_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
 --meta_data_file_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
 --fine_tuning_task_type=squad --max_seq_length=384


