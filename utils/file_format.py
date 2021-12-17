# -*-coding:utf-8-*-
from utils.tools import align_tokens_labels


def col2row(filepath):
    if 'col' in filepath:
        output_path = filepath.replace('col', 'row')
    else:
        output_path = filepath + '_row'
    fout = open(output_path, 'w', encoding='utf8')
    examples = []

    text, labels = [[] for _ in range(2)]
    for line in open(filepath, 'r', encoding='utf8'):
        if len(line[:-1].split()) > 1:
            token, label = line[:-1].split()
            text.append(token)
            labels.append(label)
        else:
            align_tokens_labels(text, labels)
            examples.append(' '.join(text) + '\t' + ' '.join(labels))
            text, labels = [[] for _ in range(2)]
    if text and labels:
        align_tokens_labels(text, labels)
        examples.append(' '.join(text) + '\t' + ' '.join(labels))
    fout.write('\n'.join(examples) + '\n')


if __name__ == '__main__':
    for dataset in ['weibo', 'resume', 'ontonotes4', 'msra']:
        filepath = f'../data/{dataset}/test.txt'
        col2row(filepath)
