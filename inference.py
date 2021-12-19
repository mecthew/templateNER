from utils.utils_metrics import (get_entities_bio, f1_score, classification_report, precision_score,
                                 recall_score, prf1_score_with_entity_length)
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer
import torch
import time
import math
import argparse
from dataset_processor import InputExample, dataset_category2template, align_tokens_labels
from dataset_processor import construct_candidate_spans_by_jieba


def template_entity(model, word_spans, input_txt, start, category2template, tokenizer, is_chinese,
                    device):
    # print(word_spans)
    # input text -> template
    word_spans_number = len(word_spans)
    num_template = len(category2template)
    input_txt = [input_txt] * (num_template * word_spans_number)

    input_ids = tokenizer(input_txt, return_tensors='pt', padding="longest")['input_ids']
    entity_dict = {idx: key for idx, key in enumerate(category2template.keys())}
    template_list = list(category2template[entity_dict[idx]] for idx in range(num_template))

    temp_list = []
    for i in range(word_spans_number):
        for j in range(num_template):
            temp_list.append(word_spans[i] + " " + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding="longest", truncation=True)['input_ids']
    # output_ids[:, 0] = tokenizer.bos_token_id if isinstance(tokenizer, BartTokenizer) \
    #     else tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    output_length_list = [0] * num_template * word_spans_number
    output_span_length = [0] * num_template * word_spans_number
    template_length = [tokenizer(template_list[idx],
                                 padding="longest",
                                 return_tensors='pt')['input_ids'].shape[1] - 2
                       for idx in range(num_template)]
    first_key = list(category2template.keys())[0]
    same_suffix_length = 1  # at least 1 because eos
    ptokens, qtokens = category2template[first_key].split(), category2template['O'].split()
    pidx, qidx = len(ptokens) - 1, len(qtokens) - 1
    while pidx >= 0 and qidx >= 0 and ptokens[pidx] == qtokens[qidx]:
        same_suffix_length += 1
        pidx -= 1
        qidx -= 1
    same_suffix_length += 1
    # print("same suffix:", same_suffix_length)
    for i in range(word_spans_number):
        span_length = (tokenizer(word_spans[i],
                                 return_tensors='pt',
                                 padding=True,
                                 truncation=True)['input_ids']).shape[1] - 2
        for j in range(num_template):
            # base_length = (tokenizer(temp_list[i * num_template + j],
            #                          return_tensors='pt',
            #                          padding=True,
            #                          truncation=True)['input_ids']).shape[1]
            output_length_list[i * num_template + j] = span_length + template_length[j] + 1 - same_suffix_length
            output_span_length[i * num_template + j] = span_length
    # print(output_length_list[:2 * num_template])
    # print(output_ids[:2 * num_template])

    score = [0.] * num_template * word_spans_number
    with torch.no_grad():
        output_ids = output_ids[:, :-1]
        output = model(input_ids=input_ids.to(device),
                       decoder_input_ids=output_ids
                       .to(device))[0]
        for i in range(output_ids.shape[1] - same_suffix_length):
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(num_template * word_spans_number):
                if i < output_length_list[j]:
                    score[j] += math.log(logits[j, int(output_ids[j][i + 1])])
    # for j in range(word_spans_number * num_template):
    #     score[j] /= output_length_list[j]
    end = start + len(word_spans[score.index(max(score)) // num_template].split()) - 1
    # return [start_index,end_index,label,score]
    return [start, end, entity_dict[(score.index(max(score)) % num_template)], max(score)]


def prediction(model, input_txt, category2template, tokenizer, is_chinese, device):
    input_txt_list = input_txt.split(' ')

    entity_list = []
    span_max_len = args.span_max_len + int(len(input_txt_list) * args.span_alpha)
    delimiter = '' if is_chinese else ' '
    candidate_spans = construct_candidate_spans_by_jieba(input_txt_list, max_span_len=span_max_len,
                                                         delimiter=delimiter, min_cover_len=3,
                                                         is_chinese=is_chinese)
    candidate_spans = [[' '.join(input_txt_list[pos[0]: pos[1] + 1]) for pos in candidate_spans if pos[0] == i]
                       for i in range(len(input_txt_list))]
    # for i in range(len(input_txt_list)):
    #     word_spans = []
    #     span_max_len = args.span_max_len + int(len(input_txt_list) * args.span_alpha)
    #     for j in range(1, span_max_len + 1):
    #         if i+j > len(input_txt_list) or input_txt_list[i+j-1] in punctuations:
    #             break
    #         word_span = ' '.join(input_txt_list[i:i + j])
    #         word_spans.append(word_span)
    for ith, word_spans in enumerate(candidate_spans):
        if len(word_spans) > 0:
            word_spans.sort(key=lambda x: len(x.split()))
            # [start_index,end_index,label,score]
            entity = template_entity(model, word_spans, input_txt, ith, category2template, tokenizer, is_chinese,
                                     device)
            if entity[1] >= len(input_txt_list):
                entity[1] = len(input_txt_list) - 1
            if entity[2] != 'O':
                entity_list.append(entity)

    # Remove entities with position conflicts
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i + 1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * len(input_txt_list)

    for entity in entity_list:
        label_list[entity[0]:entity[1] + 1] = ["I-" + entity[2]] * (entity[1] - entity[0] + 1)
        label_list[entity[0]] = "B-" + entity[2]
    return label_list


def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def inference(args):
    is_chinese = True if args.dataset in ['msra', 'ontonotes4', 'resume', 'weibo'] else False
    category2template = dataset_category2template[args.dataset]
    if is_chinese:
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
    else:
        tokenizer = BartTokenizer.from_pretrained(args.pretrain_path)
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    model.eval()
    model.config.use_cache = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    file_path = f'./data/{args.dataset}/test.txt'
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    if is_chinese:
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
            if is_chinese:
                words, labels = align_tokens_labels(words, labels)
            examples.append(InputExample(words=words, labels=labels))

    trues_list = []
    preds_list = []
    delimeter = ' '
    num_01 = len(examples)
    num_point = 0
    start = time.time()
    for example in examples:
        sources = delimeter.join(example.words)
        preds_list.append(prediction(model, sources, category2template, tokenizer, is_chinese, device))
        trues_list.append(example.labels)
        print('%d/%d (%s)' % (num_point + 1, num_01, cal_time(start)))
        print('Pred:', preds_list[num_point])
        print('Gold:', trues_list[num_point])
        num_point += 1

    true_entities = get_entities_bio(trues_list)
    pred_entities = get_entities_bio(preds_list)
    results = {
        "f1": f1_score(true_entities, pred_entities),
        "precision": precision_score(true_entities, pred_entities),
        "recall": recall_score(true_entities, pred_entities),
        "entity_length_prf1": prf1_score_with_entity_length(true_entities, pred_entities)
    }
    print(results)
    print(classification_report(true_entities, pred_entities))
    return results
    # save preds_list and trues_list
    # for num_point in range(len(preds_list)):
    #     preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
    #     trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
    # with open('./pred.txt', 'w') as f0:
    #     f0.writelines(preds_list)
    # with open('./gold.txt', 'w') as f0:
    #     f0.writelines(trues_list)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='conll2003')
parser.add_argument("--pretrain_path", type=str, default="")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--span_max_len", type=int, default=10)
parser.add_argument('--span_alpha', type=float, default=0.0)
args = parser.parse_args()
print(args.checkpoint)
inference(args)
