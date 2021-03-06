
export COQA_DIR=../datasets/CoQA
export BERT_BASE_DIR=../pretrained/uncased_L-12_H-768_A-12
export MODEL_DIR=../outputs/coqa_bert_span/


python ../run_coqa_bert_span.py \
  --input_meta_data_path=${COQA_DIR}/coqa_bert_span_meta_data \
  --train_data_path=${COQA_DIR}/coqa_bert_span_train.tf_record \
  --predict_file=${COQA_DIR}/coqa-dev-v1.0.sample.json \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=4 \
  --predict_batch_size=4 \
  --learning_rate=8e-5 \
  --num_train_epochs=2 \
  --model_dir=${MODEL_DIR} \
  --strategy_type=mirror \
   --model_export_path =${MODEL_DIR}/saved
