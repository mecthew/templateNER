# -*-coding:utf-8-*-
import json
import os
import random

import numpy as np
from collections import namedtuple, defaultdict
import jieba.posseg as pseg
from typing import List
from functools import reduce
import torch
from transformers import BartTokenizer, BertTokenizer
from transformers import BartForConditionalGeneration, BertForMaskedLM, BartForCausalLM

from utils_metrics import get_entities_bio
import math
import pandas as pd
from utils import normalize_tokens, align_tokens_labels, cut_sent, pos_tagging

dataset_category2template = {
    "conll2003": {
        "LOC": "is a location entity .",
        "PER": "is a person entity .",
        "ORG": "is an organization entity .",
        "MISC": "is an other entity .",
        "O": "is not a named entity ."
    },
    "msra": {
        "NR": "是 一 个 人 名 实 体 。",
        "NS": "是 一 个 地 理 实 体 。",
        "NT": "是 一 个 机 构 实 体 。",
        "O": "不 是 一 个 实 体 。",
    },
    "ontonotes4": {
        "GPE": "是 国 家 城 市 实 体 。",
        "LOC": "是 地 区 实 体 。",
        "ORG": "是 机 构 实 体 。",
        "PER": "是 人 物 实 体 。",
        "O": "不 是 实 体 。",
    },
    "resume": {
        "CONT": "是 国 家 实 体 。",
        "EDU": "是 教 育 实 体 。",
        "LOC": "是 地 区 实 体 。",
        "NAME": "是 人 名 实 体 。",
        "ORG": "是 机 构 实 体 。",
        "PRO": "是 专 业 实 体 。",
        "RACE": "是 民 族 实 体 。",
        "TITLE": "是 职 位 实 体 。",
        "O": "不 是 实 体 。",
    },
    "weibo": {
        "GPE.NAM": "是 国 家 实 体 。",
        "GPE.NOM": "指 代 城 市 实 体 。",
        "LOC.NAM": "是 景 物 实 体 。",
        "LOC.NOM": "指 代 景 物 实 体 。",
        "ORG.NAM": "是 机 构 实 体 。",
        "ORG.NOM": "指 代 机 构 实 体 。",
        "PER.NAM": "是 人 名 实 体 。",
        "PER.NOM": "指 代 人 名 实 体 。",
        "O": "不 是 实 体 。",
    },
}

punctuations = [",", ".", ";", ":", "!", "?", "\"", "，", "。", "；", "：", "？", "！", "@", "\\", "/", "#",
                "%", "&", "|", "*", "%", "~", "+", "$", "[", "]", "{", "}", "^", "☀", "�"]


class InputExample(object):
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels


class DatasetProcessor(object):

    def __init__(self, dataset_dir, category2template, span_max_len=10, span_alpha=0.05, tokenizer=None,
                 analyze_sent_cuts=False):
        self.dataset_dir = dataset_dir
        self.dataset = dataset_dir.replace('\\', '/').rsplit('/', maxsplit=1)[-1]
        self.is_chinese = True if self.dataset in ['msra', 'ontonotes4', 'resume', 'weibo'] else None
        self.category2template = category2template
        self.span_max_len = span_max_len
        self.span_alpha = span_alpha
        self.tokenizer = tokenizer
        # self.lac = paddlehub.Module(name='lac')
        self.entity_hits = defaultdict(int)
        self.output_dir = './dataset_info'
        self.is_chinese = bool(self.dataset in ['msra', 'ontonotes4', 'resume', 'weibo'])
        self.delimiter = "" if self.is_chinese else " "
        self.analyze_sent_cuts = analyze_sent_cuts

        self.train_path = os.path.join(dataset_dir, 'train.txt')
        self.dev_path = os.path.join(dataset_dir, 'valid.txt')
        self.test_path = os.path.join(dataset_dir, 'test.txt')
        if not os.path.exists(self.dev_path):
            self.dev_path = os.path.join(dataset_dir, 'dev.txt')

        self.train_data, self.dev_data, self.test_data = self._read_data()
        # index 0 means length 1
        self.dataset_entity_info = self.get_dataset_entity_info()

    def get_dataset_entity_info(self):
        settype2data = {'train': self.train_data, 'dev': self.dev_data, 'test': self.test_data}
        dataset_entity_info = defaultdict(dict)
        if self.tokenizer is not None:
            unk_token = '[UNK]' if self.is_chinese else '<unk>'
            unk_token_id = self.tokenizer.convert_tokens_to_ids([unk_token])

        unk_tokens = set()
        entities_set = defaultdict(set)
        for set_type, data in settype2data.items():
            entity_length_cnt = defaultdict(int)
            entity_tag_cnt = defaultdict(int)
            cut_hits, exactly_match = 0, 0
            incorrect_split_sent = []
            incorrect_split_cnt = defaultdict(int)
            for example in data:
                if self.tokenizer is not None:
                    try:
                        token_ids = self.tokenizer(example.words,
                                                   return_tensors='np',
                                                   add_special_tokens=False)['input_ids']
                        for idx in np.argwhere(token_ids == unk_token_id):
                            unk_tokens.add(example.words[idx[0]])
                    except Exception as e:
                        print(example.words, example.labels, e)
                        continue

                words, labels = example.words, example.labels
                entities = get_entities_bio(labels)
                for entity in entities:
                    entity_type, posb, pose = entity
                    entity_length_cnt[pose - posb + 1] += 1
                    entities_set[set_type].add(self.delimiter.join(words[posb: pose + 1]))

                if self.analyze_sent_cuts:
                    cuts_with_char_pos = cut_sent(''.join(words), cut_all=True)
                    entities_with_char_pos = [(ent[0], sum(map(len, words[:ent[1]])),
                                               sum(map(len, words[:ent[1]])) + len(
                                                   ''.join(words[ent[1]: ent[2] + 1])) - 1,
                                               self.delimiter.join(words[ent[1]: ent[2] + 1])) for ent in entities]
                    entities_with_char_pos = sorted(entities_with_char_pos, key=lambda x: (x[1], x[2]))
                    cut_idx = 0
                    incorrect_split_entities = list()
                    for ent in entities_with_char_pos:
                        ent_type, posb, pose, ent_name = ent
                        while cut_idx < len(cuts_with_char_pos) and cuts_with_char_pos[cut_idx][-1] < pose:
                            cut_idx += 1

                        if cut_idx == len(cuts_with_char_pos):
                            print(ent, ''.join(words), cuts_with_char_pos)
                        cuts_with_same_end = [cuts_with_char_pos[cut_idx]]
                        cut_end = cuts_with_char_pos[cut_idx][-1]
                        _cut_idx = cut_idx + 1
                        while _cut_idx < len(cuts_with_char_pos) and cuts_with_char_pos[_cut_idx][-1] == cut_end:
                            cuts_with_same_end.append(cuts_with_char_pos[_cut_idx])
                            _cut_idx += 1

                        if pose == cuts_with_char_pos[cut_idx][-1] and posb in [c[1] for c in cuts_with_same_end]:
                            exactly_match += 1
                            cut_hits += 1
                        elif pose == cuts_with_char_pos[cut_idx][-1]:
                            cut_hits += 1
                        else:
                            incorrect_split_cnt[pose - posb + 1] += 1
                            if cut_idx > 0:
                                incorrect_split_entities.append([ent, cuts_with_char_pos[cut_idx - 1],
                                                                 cuts_with_char_pos[cut_idx]])
                            else:
                                incorrect_split_entities.append([ent, [],
                                                                 cuts_with_char_pos[cut_idx]])
                    if len(incorrect_split_entities) > 0:
                        incorrect_split_sent.append((incorrect_split_entities, self.delimiter.join(words)))

            entity_length_array = [0] * max(entity_length_cnt.keys()) if len(data) > 0 else []
            for k, v in entity_length_cnt.items():
                entity_length_array[k - 1] += v
            dataset_entity_info[set_type]['entity_length_array'] = entity_length_array
            dataset_entity_info[set_type]['entity_longest_len'] = len(entity_length_array)
            dataset_entity_info[set_type]['total_entities_num'] = sum(entity_length_array)

            if self.analyze_sent_cuts:
                entity_tag_cnt = {k: v for k, v in sorted(entity_tag_cnt.items(), key=lambda x: x[1], reverse=True)}
                incorrect_split_sent = '\n'.join(list(map(str, incorrect_split_sent[:50])))
                total_entity_num = dataset_entity_info[set_type]['total_entities_num']
                print(
                    f"{set_type}: exactly match rate {exactly_match / total_entity_num};"
                    f" append len<3 entity:{(exactly_match + sum(incorrect_split_cnt[l] for l in range(1, 4))) / total_entity_num}\n"
                    f"cut sentence hit rate {cut_hits / total_entity_num};"
                    f" append len<3 entity: {(cut_hits + sum(incorrect_split_cnt[l] for l in range(1, 4))) / total_entity_num}")
                print(f"{set_type}: \nincorrect split: {incorrect_split_sent}\nentity_tag_cnt: {entity_tag_cnt}")

        print(
            f"Train_Dev intersection rate: "
            f"{len(entities_set['train'].intersection(entities_set['dev'])) / (len(entities_set['dev']) + 1e-9)}, "
            f"Train_Test intersection rate: "
            f"{len(entities_set['train'].intersection(entities_set['test'])) / (len(entities_set['test']) + 1e-9)}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, f'{self.dataset}_entities.txt'), 'w', encoding='utf8') as fout:
            total_entities_set = set()
            for set_type_entities in entities_set.values():
                total_entities_set = total_entities_set.union(set_type_entities)
            fout.write('\n'.join(list(total_entities_set)))
        print(json.dumps(dataset_entity_info, ensure_ascii=False))
        print("unknown tokens:", unk_tokens)
        return dataset_entity_info

    def dataset_summary(self):
        categories = set(label[2:] for example in self.train_data for label in example.labels if len(label) > 2)
        train_lens = sorted(list(map(len, [example.words for example in self.train_data])))
        dev_lens = sorted(list(map(len, [example.words for example in self.dev_data])))
        test_lens = sorted(list(map(len, [example.words for example in self.test_data])))
        train_entity_lengths = self.dataset_entity_info['train']['entity_length_array']
        total_train_entity_num = self.dataset_entity_info['train']['total_entities_num']
        train_entity_995percent_idx = 0
        train_entity_995percent_num = 0
        while train_entity_995percent_num < int(0.995 * total_train_entity_num) \
                and train_entity_995percent_idx < len(train_entity_lengths):
            train_entity_995percent_num += train_entity_lengths[train_entity_995percent_idx]
            train_entity_995percent_idx += 1
        dataset_info = {
            "train_examples": len(self.train_data),
            "dev_examples": len(self.dev_data),
            "test_examples": len(self.test_data),
            "categories": sorted(list(categories)),
            "train_entity_max_len": len(train_entity_lengths),
            "train_entity_99.5%_len": train_entity_995percent_idx + 1,
            "train_sent_avg_len": round(np.mean(train_lens), 2),
            "train_sent_99.9%_len": train_lens[int(len(train_lens) * 0.999)],
            "dev_sent_avg_len": round(np.mean(dev_lens), 2) if len(dev_lens) else "null",
            "dev_sent_99.9%_len": dev_lens[int(len(dev_lens) * 0.999)] if len(dev_lens) else "null",
            "test_sent_avg_len": round(np.mean(test_lens), 2) if len(dev_lens) else "null",
            "test_sent_99.9%_len": test_lens[int(len(test_lens) * 0.999)] if len(dev_lens) else "null",
        }
        print(dataset_info)
        return dataset_info

    def _read_data(self):
        train_dev_test_data = []
        for ith, filepath in enumerate([self.train_path, self.dev_path, self.test_path]):
            examples = []
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    words = []
                    labels = []
                    for line in f:
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n" \
                                or line.startswith('-------------------------'):
                            if words:
                                if self.is_chinese:
                                    words, labels = align_tokens_labels(words, labels)
                                else:
                                    words = normalize_tokens(words)
                                if len(words) > 0 and len(labels) > 0:
                                    examples.append(InputExample(words=words, labels=labels))
                                words = []
                                labels = []
                        else:
                            splits = line.split(" ")
                            words.append(splits[0])
                            if len(splits) > 1:
                                labels.append(splits[-1].replace("\n", ""))
                            else:
                                # Examples could have no label for mode = "test"
                                labels.append("O")
                    if words:
                        if self.is_chinese:
                            words, labels = align_tokens_labels(words, labels)
                        else:
                            words = normalize_tokens(words)
                        if len(words) > 0 and len(labels) > 0:
                            examples.append(InputExample(words=words, labels=labels))
            except Exception as e:
                print(f"{filepath}: {e}")
            train_dev_test_data.append(examples)
        return train_dev_test_data

    def analyze_candidate_span(self, model, tokenizer):
        os.makedirs(self.output_dir, exist_ok=True)
        settype2data = {'train': self.train_data, 'dev': self.dev_data, 'test': self.test_data} if dataset != 'msra' \
            else {'train': self.train_data, 'test': self.test_data}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for set_type, data in settype2data.items():
            if not data:
                continue
            else:
                output_file_path = os.path.join(self.output_dir, f'{self.dataset}_{set_type}_lm_logits')
                result = []
                total_entities_num, frequent_ent_num = 0, 0
                batch_size = 100
                num_iterations = math.ceil(len(data) // batch_size)
                iteration = 0
                model.eval()
                while iteration < num_iterations:
                    iteration += 1
                    examples = data[iteration * batch_size: (iteration + 1) * batch_size]
                    sentences, labels_list = [self.delimiter.join(ex.words) for ex in examples], \
                                             [ex.labels for ex in examples]
                    entities_list = [[(*ent, self.delimiter.join(ex.words[ent[1]: ent[2] + 1])) for ent in \
                                      get_entities_bio(ex.labels)] for ex in examples]
                    total_entities_num += sum(map(len, entities_list))
                    batch = tokenizer(sentences, padding='longest', return_tensors='pt')
                    batch_seq_len = torch.sum(batch['attention_mask'], dim=-1)
                    max_seq_len = batch['attention_mask'].shape[1]
                    if isinstance(model, BertForMaskedLM):
                        mask = []
                        for seq_len in batch_seq_len:
                            temp = torch.eye(max_seq_len, max_seq_len).ne(1)
                            temp[seq_len:, seq_len:] = 0
                            mask.append(temp.tolist())
                        batch['attention_mask'] = torch.tensor(mask, dtype=torch.uint8)
                    else:
                        del batch['token_type_ids']
                    if torch.cuda.is_available():
                        model = model.to(device)
                        batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.no_grad():
                        outputs = model(**batch)
                    input_ids = batch["input_ids"].cpu()
                    logits = outputs.logits.softmax(dim=-1).cpu()
                    batch_seq_len = batch_seq_len.cpu()
                    if isinstance(model, BertForMaskedLM):
                        scores = logits.gather(dim=-1, index=input_ids.unsqueeze(-1))[:, 1:, :].squeeze()
                    else:
                        scores = logits.gather(dim=-1, index=input_ids.unsqueeze(-1)[:, 1:, :]).squeeze()
                    for ith, (seq_len, entities) in enumerate(zip(batch_seq_len, entities_list)):
                        ent_scores = []
                        avg_score_for_each_token = torch.mean(scores[ith, :seq_len - 1])
                        for ent in entities:
                            ent_type, posb, pose, ent_name = ent
                            cscore = torch.mean(scores[ith, posb: pose + 1]).item()
                            frequent_ent_num += int(cscore > avg_score_for_each_token)
                    score_cut_by_seqlen = [s[:seq_len - 2] for s, seq_len in zip(scores, batch_seq_len)]
                    for ex, entities, score in zip(examples, entities_list, score_cut_by_seqlen):
                        result.append((ex.words, entities, score))
                print(f'{self.dataset} {set_type} frequent entities rate: {frequent_ent_num / total_entities_num}')
                with open(output_file_path, 'w', encoding='utf8') as fout:
                    fout.write(
                        f'{self.dataset} {set_type} frequent entities rate: {frequent_ent_num / total_entities_num}\n')
                    for items in result:
                        words, entities, score = items
                        fout.write(' '.join(words) + '\t' + str(entities) + '\t' + str(score) + '\n')

    def construct_templates_from_example(self, example, times_of_negative_example, set_type,
                                         negative_min_num=5, sentence_negative_alpha=0.1):
        CsvInputExample = namedtuple("CsvInputExample", ['input_text', "target_text"])
        span_max_len = min(self.span_max_len + int(self.span_alpha * len(example.words)),
                           self.dataset_entity_info['train']['entity_longest_len'])
        positive_target_texts = []
        negative_target_texts = []
        entities = get_entities_bio(example.labels)
        entity_positions = [(posb, pose) for ent_type, posb, pose in entities]
        for entity in entities:
            ent_type, posb, pose = entity
            positive_target_texts.append(' '.join(example.words[posb: pose + 1])
                                         + " " + self.category2template[ent_type.upper()])

        choose_probs = [cnt for cnt in self.dataset_entity_info['train']['entity_length_array'][:span_max_len]]
        choose_probs = [cnt / sum(choose_probs) for cnt in choose_probs]
        assert len(choose_probs) == span_max_len
        negative_example_probs = []
        negative_example_span_lengths = [0] * span_max_len
        candidate_spans = construct_heuristic_candidate_spans(example.words, span_max_len, delimiter=self.delimiter,
                                                              min_cover_len=3)
        candidate_spans_based_on_positive = construct_candidate_spans_based_on_positive_spans(example,
                                                                                              span_max_len,
                                                                                              negative_num=5)

        for span in candidate_spans:
            posb, pose = span
            span_len = pose - posb + 1
            if (posb, pose) in entity_positions:
                self.entity_hits[set_type] += 1
            else:
                word_span = example.words[posb: pose + 1]
                negative_target_texts.append(" ".join(word_span) + " " + self.category2template["O"])
                negative_example_probs.append((choose_probs[span_len - 1], span_len - 1))
                negative_example_span_lengths[span_len - 1] += 1

        if len(negative_target_texts) > 0:
            # normalized probability
            negative_example_probs = [item[0] / negative_example_span_lengths[item[1]] for item in
                                      negative_example_probs]
            # to avoid error caused by np.random.choice
            negative_example_probs[-1] = 1 - sum(negative_example_probs[:-1])
            negative_min_num = max(int(len(example.words) * sentence_negative_alpha), negative_min_num)
            select_negative_texts = list(
                np.random.choice(negative_target_texts,
                                 replace=False,
                                 size=min(negative_min_num, len(negative_target_texts)),
                                 p=negative_example_probs
                                 ))

            # append bias negative sample
            for negative_span in candidate_spans_based_on_positive:
                posb, pose = negative_span
                negative_text = " ".join(example.words[posb: pose+1]) + " " + self.category2template["O"]
                select_negative_texts.append(negative_text)
            select_negative_texts = list(set(select_negative_texts))
        else:
            select_negative_texts = []
            print("null negative examples", example.words)

        template_examples = []
        for target_text in positive_target_texts + select_negative_texts:
            template_example = CsvInputExample(input_text=' '.join(example.words),
                                               target_text=target_text)
            template_examples.append(template_example)
        return template_examples, positive_target_texts, negative_target_texts

    def to_csv(self, times_of_negative_example=1.5):
        train_dev_test_data = {'train': self.train_data, "dev": self.dev_data,
                               "test": self.test_data}

        for set_type, examples in train_dev_test_data.items():
            output_path = os.path.join(self.dataset_dir, f"{set_type}.csv")
            set_type_template_examples = []
            for example in examples:
                new_times_of_negative_example = times_of_negative_example if set_type == 'train' else 1.0
                template_examples, _, _ = self.construct_templates_from_example(example,
                                                                                new_times_of_negative_example,
                                                                                set_type)
                set_type_template_examples.extend(template_examples)
            dataframe = pd.DataFrame({'input_text': [ex.input_text for ex in set_type_template_examples],
                                      "target_text": [ex.target_text for ex in set_type_template_examples]})
            # if set_type in ['train', 'dev']:
            #     dataframe.to_csv(output_path, index=False, sep=',', header=True, encoding='utf8')
            print(f"{set_type}: "
                  f"{self.entity_hits[set_type] / self.dataset_entity_info[set_type]['total_entities_num']}")
            print(f"{set_type.capitalize()} data save to csv: {output_path}")


def construct_heuristic_candidate_spans(words, max_span_len, delimiter='', min_cover_len=3, is_chinese=True):
    """
    :param example: a namedtuple object, contains keys `words` and `labels`
    :param max_span_len: the max length of candidate spans
    :param delimiter: str
    :param min_cover_len: spans that lengths not larger than `min_cover_len` would be preserved
    :param is_chinese: bool
     :return: List, [(span_begin1, span_end1), ...]
    """
    sentence = delimiter.join(words)
    cuts_with_char_pos = cut_sent(sentence, cut_all=True)
    cuts_start_pos = set([cut[1] for cut in cuts_with_char_pos])
    cuts_end_pos = set([cut[2] for cut in cuts_with_char_pos])
    wordidx2sentidx = {i: (sum(map(len, words[:i])), sum(map(len, words[:i + 1])) - 1) for i in range(len(words))}

    candidate_spans = []
    for i in range(len(words)):
        for j in range(1, max_span_len + 1):
            if i + j > len(words) or words[i + j - 1] in punctuations:
                break
            # the part of speech tagging of the first word can not in the list
            # elif is_chinese and pos_tagging(delimiter.join(words[i: i + j]))[0][-1] \
            #         in ['p', 'yg', 'ag', 'x', 'y', 'h', 'f', 'c']:
            #     break
            elif j <= min_cover_len:
                candidate_spans.append((i, i + j - 1))
            elif wordidx2sentidx[i][0] in cuts_start_pos and wordidx2sentidx[i + j - 1][1] in cuts_end_pos:
                candidate_spans.append((i, i + j - 1))
            else:
                pass  # filter spans
    return candidate_spans


def construct_candidate_spans_based_on_positive_spans(example, max_span_len, negative_num=6):
    entities = get_entities_bio(example.labels)
    candidate_spans = set()
    for entity in entities:
        entity_type, posb, pose = entity
        for _ in range(negative_num):
            i = random.randint(-2, 2)
            j = random.randint(-2, 2)
            if 0 <= posb + i <= pose + j < len(example.words) and (i != 0 or j != 0) \
                    and pose + j - posb - i + 1 <= max_span_len \
                    and all(w not in punctuations for w in example.words[posb+i: pose+j+1]):
                candidate_spans.add((posb + i, pose + j))
    return candidate_spans


if __name__ == '__main__':
    import platform

    dataset = 'resume'
    if 'Windows' in platform.platform():
        # pretrain_path = r'D:\Documents\Github2021\FAN\pretrained_model\facebook-bart-base' \
        #     if dataset in ['conll2003'] else r'D:\Documents\Github2021\FAN\pretrained_model\fnlp-bart-base-chinese'
        pretrain_path = r'D:\Documents\Github2021\FAN\pretrained_model\hfl-chinese-bert-wwm-ext'
    else:
        # pretrain_path = r'/home/liujian/NLP/corpus/transformers/facebook-bart-base' \
        #     if dataset in ['conll2003'] else r'/home/liujian/NLP/corpus/transformers/fnlp-bart-base-chinese'
        pretrain_path = r'/home/qiumengchuan/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext'
    tokenizer = BartTokenizer.from_pretrained(pretrain_path) if dataset in ['conll2003'] \
        else BertTokenizer.from_pretrained(pretrain_path)
    if 'bert' in pretrain_path:
        model = BertForMaskedLM.from_pretrained(pretrain_path)
    else:
        model = BartForCausalLM.from_pretrained(pretrain_path, add_cross_attention=False)

    times_of_negative_example = 1.5 if dataset in ['msra', 'ontonotes4'] else 2.0
    dataset2span_params = {'conll2003': (10, 0.0),
                           "weibo": (8, 0.0),
                           "msra": (15, 0.0),
                           "ontonotes4": (15, 0.0),
                           "resume": (20, 0.0)}
    dataset_processor = DatasetProcessor(dataset_dir=f'./data/{dataset}',
                                         category2template=dataset_category2template[dataset],
                                         span_max_len=dataset2span_params[dataset][0],
                                         span_alpha=dataset2span_params[dataset][1],
                                         tokenizer=None,
                                         analyze_sent_cuts=False)
    dataset_processor.dataset_summary()
    dataset_processor.to_csv(times_of_negative_example=times_of_negative_example)
    # dataset_processor.analyze_candidate_span(model, tokenizer)

    # tokenizer = BartTokenizer.from_pretrained(r'D:\Documents\Github2021\FAN\pretrained_model\facebook-bart-base')
    # temp_list = [' '.join(ex.words) for ex in dataset_processor.train_data[:3]]
    # output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # print(output_ids)
    # text = "️w e i x z ÏÖÜŸａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢ ⒌ ⒐".split()
    # print(normalize_tokens(text))
