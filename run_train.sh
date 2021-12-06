if [ $1 == conll2003 ]; then
  max_seq_length=50
  num_train_epochs=10
  target_max_length=25
  train_batch_size=64
  encoder_decoder_type=bart
  pretrain_path=/home/liujian/NLP/corpus/transformers/facebook-bart-large
elif [ $1 == weibo ]; then
  max_seq_length=150
  num_train_epochs=10
  target_max_length=25
  train_batch_size=64
  encoder_decoder_type=chinese-bart
  pretrain_path=/home/liujian/NLP/corpus/transformers/fnlp-bart-base-chinese
elif [ $1 == resume ]; then
  max_seq_length=180
  num_train_epochs=10
  target_max_length=30
  train_batch_size=64
  encoder_decoder_type=chinese-bart
  pretrain_path=/home/liujian/NLP/corpus/transformers/fnlp-bart-base-chinese
elif [ $1 == ontonotes4 ]; then
  max_seq_length=240
  num_train_epochs=5
  target_max_length=30
  train_batch_size=64
  encoder_decoder_type=chinese-bart
  pretrain_path=/home/liujian/NLP/corpus/transformers/fnlp-bart-base-chinese
elif [ $1 == msra ]; then
  max_seq_length=240
  num_train_epochs=5
  target_max_length=30
  train_batch_size=64
  encoder_decoder_type=chinese-bart
  pretrain_path=/home/liujian/NLP/corpus/transformers/fnlp-bart-base-chinese
fi

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=$2 python train.py \
    --dataset $1 \
    --encoder_decoder_type $encoder_decoder_type \
    --pretrain_path $pretrain_path \
    --max_seq_length $max_seq_length \
    --train_batch_size $train_batch_size \
    --learning_rate 1e-5 \
    --num_train_epochs $num_train_epochs \
    --target_max_length $target_max_length \
    --manual_seed 4 \
    --save_model_steps 11898 \
    --wandb_project templateNER
