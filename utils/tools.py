# -*-coding:utf-8-*-
import random
from collections import defaultdict

import jieba
from typing import List

import torch
import warnings

from utils.langconv import tradition2simple
import jieba.posseg as pseg


class StructDict(dict):

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


def append_index_for_cuts(sent, cuts):
    cuts_pos = [list() for i in range(len(cuts))]
    cidx, ith = 0, 0
    while ith < len(cuts):
        cut = cuts[ith]
        if 0 < cidx < len(sent) and cuts[ith - 1].startswith(cut):  # for duplicate cuts, like "。","。","。"
            cidx += 1
        while cidx < len(sent) and sent[cidx] != cut[0]:
            cidx += 1

        next_idx = cidx
        cut_idx = 0
        while cut_idx < len(cut) and sent[next_idx] == cut[cut_idx]:
            next_idx += 1
            cut_idx += 1

        if cut_idx != len(cut):
            if cut == "好多好多":
                print(sent[cidx: next_idx + 1])
            cidx += 1
            if cidx >= len(sent):
                raise Exception(f"{sent} and {cuts} not matched!")
        else:
            cuts_pos[ith].append(cidx)
            cuts_pos[ith].append(next_idx - 1)
            ith += 1

    for cut, (posb, pose) in zip(cuts, cuts_pos):
        if cut != sent[posb: pose + 1]:
            print("Incorrect split", cut, sent[posb: pose + 1])
    return [(cut, *pos) for cut, pos in zip(cuts, cuts_pos)]


def cut_sent(sent, cut_all=True):
    """
    :param sent: str object
    :param cut_all: bool
    :return: return a list, each element is a tuple like `(word_span, posb, pose)`
    """
    cuts = list(jieba.cut(sent))
    cuts_with_pos = append_index_for_cuts(sent, cuts)
    if cut_all:
        cuts_all = list(jieba.cut(sent, cut_all=cut_all))
        cuts_all_with_pos = append_index_for_cuts(sent, cuts_all)
        cuts_with_pos = sorted(list(set(cuts_with_pos + cuts_all_with_pos)), key=lambda x: (x[2], -x[1]))
    return list(cuts_with_pos)


def is_english_word(word):
    for tok in word:
        if tok.lower() not in list(chr(ord('a') + i) for i in range(26)) + ['-']:
            return False
    return True


def normalize_tokens(text: str or List):
    # normalize token
    characters_translation_table = str.maketrans(
        "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
        "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUYabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    # digits_translation = {'⒈': '1.', '⒉': '2.', '⒊': '3.', '⒋': '4.', '⒌': '5.',
    #                       '⒍': '6.', '⒎': '7.', '⒏': '8.', '⒐': '9.', '⒑': '10.',
    #                       'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',
    #                       'Ⅵ': '6', 'Ⅶ': '7', 'Ⅷ': '8', 'Ⅸ': '9', 'Ⅹ': '10',
    #                       '０': '0', '１': '1', '２': '2', '３': '3', '４': '4', '５': '5',
    #                       '６': '6', '７': '7', '８': '8', '９': '9'}
    digits_chinese_translation_table = str.maketrans(
        "️．—…“”‘’＊％⒈⒉⒊⒋⒌⒍⒎⒏⒐ⅠⅡⅢⅣⅤⅥⅦⅧⅨ０１２３４５６７８９",
        " .-。\"\"\'\'%火1234567891234567890123456789"
    )

    if isinstance(text, str):
        text = text.translate(digits_chinese_translation_table)
        return text.translate(characters_translation_table)
    elif isinstance(text, List):
        # new_text = []
        # for tok in text:
        #     for key, value in digits_translation.items():
        #         tok = tok.replace(key, value)
        #     new_text.append(tok)
        new_text = [tok.translate(digits_chinese_translation_table) for tok in text]
        return [tok.translate(characters_translation_table) for tok in new_text]
    else:
        raise ValueError(f"type {type(text)} is not supported !")


def align_tokens_labels(tokens: List, labels: List):
    """
        Reorganize English characters to form a word and align labels
    """
    origin_len = sum(map(len, tokens))
    origin_tokens = tokens.copy()
    tokens = normalize_tokens(tokens)
    tokens = tradition2simple(tokens)
    normalize_len = sum(map(len, tokens))
    if origin_len != normalize_len:
        warnings.warn(f"tokens length inconsistent after normalizing, origin: {origin_tokens}, {normalize_len},"
                      f" normalize: {tokens}, {origin_len}, ")

    new_tokens, new_labels = [], []
    prev_word = ""
    prev_tag_list = []
    for token, label in zip(tokens, labels):
        if token.lower() in set(chr(ord('a') + i) for i in range(26)):
            if prev_tag_list and prev_tag_list[0][0] in ['B', 'I'] and (
                    label.startswith("B-") or label.startswith('O')):
                new_tokens.append(prev_word)
                new_labels.append(prev_tag_list[0])
                prev_word = ""
                prev_tag_list = []

            prev_word += token
            prev_tag_list.append(label)
        else:
            if prev_word:
                new_tokens.append(prev_word)
                new_labels.append(prev_tag_list[0])
                prev_word = ""
                prev_tag_list = []
            if token.strip():
                new_tokens.append(token)
                new_labels.append(label)
    assert len(new_tokens) == len(new_labels)
    return new_tokens, new_labels


def pos_tagging(sentence):
    return [(w.word, w.flag) for w in pseg.cut(sentence)]


def chinese_to_english_punct(sent, dims=1, replace_lst=["，", "。", "！", "？", "；", "（", "）", "＠", "＃", "【", "】", "+", "=", "-", "：", "“",  "”",  "‘",  "’",  "》",  "《",  "「",  "」",], target_lst =  [",", ".", "!", "?", ";", "(", ")", "@", "#", "[", "]", "+", "=", "-", ":", '"', '"', "'", "'", ">", "<", "{", "}", ]):
    """

    :param sent: List[str] or str
    :param dims: dims=1 means sent is `str`, dims=2 means sent is List[str]
    :param replace_lst: chinese punctuations
    :param target_lst: normalized punctuations
    :return:
    """
    # chinese punctuation to english punctuation
    if dims == 1:
        for item_idx, (replace_item, target_item) in enumerate(zip(replace_lst, target_lst)):
            if replace_item not in sent:
                continue
            sent = sent.replace(replace_item, target_item)
        return sent
    elif dims == 2:
        tar_lst = []
        for sent_item in sent:
            tmp_sent = chinese_to_english_punct(sent_item, dims=1)
            tar_lst.append(tmp_sent)
        return tar_lst


def make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention. 1 means `no mask`, 0 means `mask`
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), 0)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.ones(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, :, :].expand(bsz, tgt_len, tgt_len + past_key_values_length)


if __name__ == '__main__':
    # tag_counter = defaultdict(int)
    # dataset = 'weibo'
    # with open(f'./dataset_info/{dataset}_entities.txt', encoding='utf8') as fin:
    #     for line in fin:
    #         if line.strip():
    #             word_tags = pos_tagging(line.strip())
    #             tag_counter[word_tags[0][1]] += 1
    #
    # tag_counter = {k: v for k, v in sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)}
    # print(tag_counter)

    for i in range(10):
        print(random.randint(-2, 3))
