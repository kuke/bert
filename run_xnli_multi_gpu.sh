export CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export BERT_BASE_DIR=chinese_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
export XNLI_DIR=xnli_data

python run_classifier_multi_gpu.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --num_gpus=1 \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=./xnli_clean_code_output
