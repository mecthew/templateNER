if [ $1 == conll2003 ]; then
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/facebook-bart-large
  checkpoint=outputs_conll2003_facebook-bart-large/best_model
  span_max_len=10
elif [ $1 == weibo ]; then
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/fnlp-bart-base-chinese
  checkpoint=outputs_weibo_fnlp-bart-base-chinese_tcode7615/best_model
  span_max_len=8
elif [ $1 == resume ]; then
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/fnlp-bart-base-chinese
  checkpoint=outputs_resume_fnlp-bart-base-chinese_tcode7615/best_model
  span_max_len=22
elif [ $1 == ontonotes4 ]; then
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/fnlp-bart-base-chinese
  checkpoint=outputs_ontonotes4_fnlp-bart-base-chinese_tcode8988/best_model
  span_max_len=15
elif [ $1 == msra ]; then
  pretrain_path=/home/qiumengchuan/NLP/corpus/transformers/fnlp-bart-base-chinese
  checkpoint=outputs_msra_fnlp-bart-base-chinese/best_model
  span_max_len=16
fi

PYTHONIOENCODING=utf8 CUDA_VISIBLE_DEVICES=$2 python inference.py \
    --dataset $1 \
    --pretrain_path $pretrain_path \
    --checkpoint $checkpoint \
    --span_max_len $span_max_len \
    --span_alpha 0.0