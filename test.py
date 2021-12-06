# -*-coding:utf-8-*-
from transformers import BertTokenizer
print("南".isalpha())
tokenizer = BertTokenizer.from_pretrained(r'D:\Documents\Github2021\FAN\pretrained_model\fnlp-bart-base-chinese')
print(tokenizer("我们去哪里呢？ok"))