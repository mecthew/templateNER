# -*-coding:utf-8-*-
import os
import numpy as np
from collections import namedtuple, defaultdict

from transformers import BartTokenizer

from utils_metrics import get_entities_bio
import math
import pandas as pd

dataset_category2template = {
    "conll2003": {
        "LOC": "is a location entity .",
        "PER": "is a person entity .",
        "ORG": "is an organization entity .",
        "MISC": "is an other entity .",
        "O": "is not a named entity ."
    },
    "msra": {
        "NR": "是 人 名 实 体 。",
        "NS": "是 地 理 实 体 。",
        "NT": "是 机 构 实 体 。",
        "O": "不 是 实 体 。",
    },
    "ontonotes4": {
        "GPE": "是 政 治 地 缘 实 体 。",
        "LOC": "是 地 理 实 体 。",
        "ORG": "是 机 构 实 体 。",
        "PER": "是 人 名 实 体 。",
        "O": "不 是 实 体 。",
    },
    "resume": {
        "CONT": "是 国 家 实 体 。",
        "EDU": "是 教 育 实 体 。",
        "LOC": "是 地 理 实 体 。",
        "NAME": "是 人 名 实 体 。",
        "ORG": "是 机 构 实 体 。",
        "PRO": "是 专 业 实 体 。",
        "RACE": "是 民 族 实 体 。",
        "TITLE": "是 职 位 实 体 。",
        "O": "不 是 实 体 。",
    },
    "weibo": {
        "GPE.NAM": "是 政 治 地 缘 实 体 。",
        "GPE.NOM": "指 代 政 治 地 缘 实 体 。",
        "LOC.NAM": "是 地 理 实 体 。",
        "LOC.NOM": "指 代 地 理 实 体 。",
        "ORG.NAM": "是 机 构 实 体 。",
        "ORG.NOM": "指 代 机 构 实 体 。",
        "PER.NAM": "是 人 名 实 体 。",
        "PER.NOM": "指 代 人 名 实 体 。",
        "O": "不 是 实 体 。",
    },
}
punctuations = [",", ".", ";", ":", "!", "?", "\"", "，", "。", "；", "：", "？", "！", "“", "”", "@", "\\", "/", "#",
                "%", "&", "|", "*", "%", "~", "+", "$", "[", "]", "{", "}", "^"]


class InputExample(object):
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels


class DatasetProcessor(object):

    def __init__(self, dataset_dir, category2template, span_max_len=10, span_alpha=0.05):
        self.dataset_dir = dataset_dir
        self.dataset = dataset_dir.replace('\\', '/').rsplit('/', maxsplit=1)[-1]
        self.is_chinese = True if self.dataset in ['msra', 'ontonotes4', 'resume', 'weibo'] else None
        self.category2template = category2template
        self.span_max_len = span_max_len
        self.span_alpha = span_alpha
        self.train_path = os.path.join(dataset_dir, 'train.txt')
        self.dev_path = os.path.join(dataset_dir, 'valid.txt')
        self.test_path = os.path.join(dataset_dir, 'test.txt')
        if not os.path.exists(self.dev_path):
            self.dev_path = os.path.join(dataset_dir, 'dev.txt')

        self.train_data, self.dev_data, self.test_data = self._read_data()

        # index 0 means length 1
        _, self.train_entity_lengths = self.get_entity_lengths_info(self.train_data)
        _, self.dev_entity_lengths = self.get_entity_lengths_info(self.dev_data)
        _, self.test_entity_lengths = self.get_entity_lengths_info(self.test_data)
        self.train_entity_longest_len = len(self.train_entity_lengths)

    def dataset_summary(self):
        categories = set(label[2:] for example in self.train_data for label in example.labels if len(label) > 2)
        train_lens = sorted(list(map(len, [example.words for example in self.train_data])))
        dev_lens = sorted(list(map(len, [example.words for example in self.dev_data])))
        test_lens = sorted(list(map(len, [example.words for example in self.test_data])))
        total_train_entity_num = sum(self.train_entity_lengths)
        train_entity_995percent_idx = 0
        train_entity_995percent_num = 0
        while train_entity_995percent_num < int(0.995 * total_train_entity_num) \
                and train_entity_995percent_idx < len(self.train_entity_lengths):
            train_entity_995percent_num += self.train_entity_lengths[train_entity_995percent_idx]
            train_entity_995percent_idx += 1
        dataset_info = {
            "train_examples": len(self.train_data),
            "dev_examples": len(self.dev_data),
            "test_examples": len(self.test_data),
            "categories": sorted(list(categories)),
            "train_entity_max_len": len(self.train_entity_lengths),
            "train_entity_99.5%_len": train_entity_995percent_idx + 1,
            "train_avg_len": round(np.mean(train_lens), 2),
            "train_99.9%_len": train_lens[int(len(train_lens) * 0.999)],
            "dev_avg_len": round(np.mean(dev_lens), 2) if len(dev_lens) else "null",
            "dev_99.9%_len": dev_lens[int(len(dev_lens) * 0.999)] if len(dev_lens) else "null",
            "test_avg_len": round(np.mean(test_lens), 2) if len(dev_lens) else "null",
            "test_99.9%_len": test_lens[int(len(test_lens) * 0.999)] if len(dev_lens) else "null",
        }
        print(dataset_info)
        return dataset_info

    @staticmethod
    def get_entity_lengths_info(examples):
        if not examples:
            return dict(), []
        entity_length_cnt = defaultdict(int)
        for example in examples:
            words, labels = example.words, example.labels
            entities = get_entities_bio(labels)
            for entity in entities:
                entity_type, posb, pose = entity
                if words[posb][0].lower() in set(chr(ord('a') + i) for i in range(26)):
                    print(words[posb: pose + 1])
                entity_length_cnt[pose - posb + 1] += 1
        entity_length_array = [0] * max(entity_length_cnt.keys())
        for k, v in entity_length_cnt.items():
            entity_length_array[k - 1] += v
        return entity_length_cnt, entity_length_array

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
                        examples.append(InputExample(words=words, labels=labels))
            except:
                print(f"{filepath} not exists")
            train_dev_test_data.append(examples)
        return train_dev_test_data

    def construct_template_from_example(self, example, times_of_negative_example):
        CsvInputExample = namedtuple("CsvInputExample", ['input_text', "target_text"])
        span_max_len = min(self.span_max_len + int(self.span_alpha * len(example.words)), self.train_entity_longest_len)
        positive_target_texts = []
        negative_target_texts = []
        entities = get_entities_bio(example.labels)
        entity_positions = [(posb, pose) for ent_type, posb, pose in entities]
        for entity in entities:
            ent_type, posb, pose = entity
            positive_target_texts.append(' '.join(example.words[posb: pose + 1])
                                         + " " + self.category2template[ent_type.upper()])

        choose_probs = [cnt for cnt in self.train_entity_lengths[:span_max_len]]
        choose_probs = [cnt / sum(choose_probs) for cnt in choose_probs]
        assert len(choose_probs) == span_max_len
        negative_example_probs = []
        negative_example_span_lengths = [0] * span_max_len
        for i in range(len(example.words)):
            for j in range(1, span_max_len + 1):
                if i + j <= len(example.words) and (i, i + j - 1) not in entity_positions:
                    if example.words[i + j - 1] in punctuations:  # entity do not contain punctuation
                        break
                    word_span = example.words[i: i + j]
                    negative_target_texts.append(" ".join(word_span) + " " + self.category2template["O"])
                    negative_example_probs.append((choose_probs[j - 1], j))
                    negative_example_span_lengths[j - 1] += 1

        if negative_target_texts:
            # normalized probability
            negative_example_probs = [item[0] / negative_example_span_lengths[item[1] - 1] for item in
                                      negative_example_probs]
            negative_example_probs[-1] = 1 - sum(negative_example_probs[:-1])
            select_negative_texts = list(
                np.random.choice(negative_target_texts,
                                 replace=False,
                                 size=min(max(math.ceil(times_of_negative_example*2),
                                              math.ceil(len(entities) * times_of_negative_example)),
                                          len(negative_target_texts)),
                                 ))
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
        train_dev_test_data = {'train': self.train_data, "dev": self.dev_data}

        for set_type, examples in train_dev_test_data.items():
            output_path = os.path.join(self.dataset_dir, f"{set_type}.csv")
            set_type_template_examples = []
            for example in examples:
                new_times_of_negative_example = times_of_negative_example if set_type == 'train' else 1.0
                template_examples, _, _ = self.construct_template_from_example(example, new_times_of_negative_example)
                set_type_template_examples.extend(template_examples)
            dataframe = pd.DataFrame({'input_text': [ex.input_text for ex in set_type_template_examples],
                                      "target_text": [ex.target_text for ex in set_type_template_examples]})
            dataframe.to_csv(output_path, index=False, sep=',', header=True, encoding='utf8')
            print(f"{set_type.capitalize()} data save to csv: {output_path}")


def align_tokens_labels(tokens, labels):
    """
        Reorganize English characters to form a word
    """
    tokens = remove_accents(tokens, dims=2)
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

            if token != '️':    # remove full space
                new_tokens.append(token)
                new_labels.append(label)
    assert len(new_tokens) == len(new_labels)
    return new_tokens, new_labels


def remove_accents(text: str, dims: int):
    accents_translation_table = str.maketrans(
        "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
        "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    if dims == 1:
        return text.translate(accents_translation_table)
    elif dims == 2:
        return [tok.translate(accents_translation_table) for tok in text]


if __name__ == '__main__':
    dataset = 'weibo'
    times_of_negative_example = 2.0 if dataset != 'weibo' else 2.0
    dataset2span_params = {'conll2003': (10, 0.0),
                           "weibo": (10, 0.0),
                           "msra": (20, 0.0),
                           "ontonotes4": (15, 0.0),
                           "resume": (20, 0.0)}
    dataset_processor = DatasetProcessor(dataset_dir=fr'D:\Documents\Github2021\templateNER\data\{dataset}',
                                         category2template=dataset_category2template[dataset],
                                         span_max_len=dataset2span_params[dataset][0],
                                         span_alpha=dataset2span_params[dataset][1])
    dataset_processor.dataset_summary()
    dataset_processor.to_csv(times_of_negative_example=times_of_negative_example)

    # tokenizer = BartTokenizer.from_pretrained(r'D:\Documents\Github2021\FAN\pretrained_model\facebook-bart-base')
    # temp_list = [' '.join(ex.words) for ex in dataset_processor.train_data[:3]]
    # output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # print(output_ids)
