
export SQUAD_DIR=../datasets/squad
export SQUAD_VERSION=v1.1
export BERT_BASE_DIR=../pretrained/uncased_L-12_H-768_A-12
export MODEL_DIR=../outputs/squad/



python ../run_squad.py \
  --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
  --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
  --predict_file=${SQUAD_DIR}/dev-v1.1.json \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=4 \
  --predict_batch_size=4 \
  --learning_rate=8e-5 \
  --num_train_epochs=2 \
  --model_dir=${MODEL_DIR} \
  --strategy_type=mirror