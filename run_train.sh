if [ $1 == conll2003 ]; then
  max_seq_length=50
  num_train_epochs=10
  target_max_length=25
  train_batch_size=64
  encoder_decoder_type=bart
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/facebook-bart-large
  span_max_len=10
  times_of_negative_example=1.5
elif [ $1 == weibo ]; then
  max_seq_length=180
  num_train_epochs=10
  target_max_length=25
  train_batch_size=64
  encoder_decoder_type=chinese-bart
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/fnlp-bart-base-chinese
  span_max_len=8
  times_of_negative_example=2.0
elif [ $1 == resume ]; then
  max_seq_length=200
  num_train_epochs=10
  target_max_length=32
  train_batch_size=64
  encoder_decoder_type=chinese-bart
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/fnlp-bart-base-chinese
  span_max_len=22
  times_of_negative_example=2.0
elif [ $1 == ontonotes4 ]; then
  max_seq_length=250
  num_train_epochs=5
  target_max_length=25
  train_batch_size=50
  encoder_decoder_type=chinese-bart
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/fnlp-bart-base-chinese
  span_max_len=15
  times_of_negative_example=1.5
elif [ $1 == msra ]; then
  max_seq_length=250
  num_train_epochs=5
  target_max_length=30
  train_batch_size=50
  encoder_decoder_type=chinese-bart
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/fnlp-bart-base-chinese
  span_max_len=20
  times_of_negative_example=1.5
fi

echo "Run dataset $1, model=$encoder_decoder_type, bz=$train_batch_size, eps=$num_train_epochs, span_max_len=$span_max_len，\
maxlen=$max_seq_length"
PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=$2 python train.py \
    --dataset $1 \
    --encoder_decoder_type $encoder_decoder_type \
    --pretrain_path $pretrain_path \
    --max_seq_length $max_seq_length \
    --train_batch_size $train_batch_size \
    --learning_rate 1e-5 \
    --num_train_epochs $num_train_epochs \
    --target_max_length $target_max_length \
    --span_max_len $span_max_len \
    --times_of_negative_example $times_of_negative_example \
    --manual_seed 4 \
    --save_model_steps 10000 \
    --wandb_project templateNER \
    --predict_on_test
