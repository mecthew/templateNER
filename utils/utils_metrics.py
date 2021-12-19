import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def get_entities(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return set(chunks)


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def get_entities_bios(seq):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


def get_entities_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


def get_entities_span(starts, ends):
    if any(isinstance(s, list) for s in starts):
        starts = [item for sublist in starts for item in sublist + ['<SEP>']]
    if any(isinstance(s, list) for s in ends):
        ends = [item for sublist in ends for item in sublist + ['<SEP>']]
    chunks = []
    for start_index, start in enumerate(starts):
        if start in ['O', '<SEP>']:
            continue
        for end_index, end in enumerate(ends[start_index:]):
            if start == end:
                chunks.append((start, start_index, start_index + end_index))
                break
            elif end == '<SEP>':
                break
    return set(chunks)


def prf1_score_with_entity_length(true_entities, pred_entities):
    """ Compute F1 score according to entity length"""
    true_pred_with_entity_length = dict()
    for ent in true_entities:
        ent_type, posb, pose = ent
        ent_len = pose - posb + 1
        if ent_len not in true_pred_with_entity_length:
            true_pred_with_entity_length[ent_len] = [{ent}, set()]
        else:
            true_pred_with_entity_length[ent_len][0].add(ent)
    for ent in pred_entities:
        ent_type, posb, pose = ent
        ent_len = pose - posb + 1
        if ent_len not in true_pred_with_entity_length:
            true_pred_with_entity_length[ent_len] = [set(), {ent}]
        else:
            true_pred_with_entity_length[ent_len][1].add(ent)

    result = dict()
    for key, true_pred_list in true_pred_with_entity_length.items():
        result[key] = (precision_score(*true_pred_list),
                       recall_score(*true_pred_list),
                       f1_score(*true_pred_list))
    return sorted(result.items(), key=lambda x: x[0])


def f1_score(true_entities, pred_entities):
    """Compute the F1 score."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def precision_score(true_entities, pred_entities):
    """Compute the precision."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(true_entities, pred_entities):
    """Compute the recall."""
    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def classification_report(true_entities, pred_entities, digits=5):
    """Build a text report showing the main classification metrics."""
    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []

    for type_name, type_true_entities in d1.items():
        type_pred_entities = d2[type_name]
        nb_correct = len(type_true_entities & type_pred_entities)
        nb_pred = len(type_pred_entities)
        nb_true = len(type_true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format('micro avg',
                             precision_score(true_entities, pred_entities),
                             recall_score(true_entities, pred_entities),
                             f1_score(true_entities, pred_entities),
                             np.sum(s),
                             width=width, digits=digits)
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report


def convert_span_to_bio(starts, ends):
    labels = []
    for start, end in zip(starts, ends):
        entities = get_entities_span(start, end)
        label = ['O'] * len(start)
        for entity in entities:
            label[entity[1]] = 'B-{}'.format(entity[0])
            label[entity[1] + 1: entity[2] + 1] = ['I-{}'.format(entity[0])] * (entity[2] - entity[1])
        labels.append(label)
    return labels


def get_type_error_rate(true_entities, pred_entities):
    """
        Get type error rate of entities that have correct boundaries
    :param true_entities: set object, contains elements like {("LOC", (1, 1)), ...}
    :param pred_entities: set object, contains elements like {("LOC", (1, 1)), ...}
    :return: tuple, type_error_rate_of_all_boundary_correct_entities, type_error_rate_of_pred_error_entities
    """
    true_entities_boundary2type = {(ent[1], ent[2]): ent[0] for ent in true_entities}
    boundary_correct_entities = [ent for ent in pred_entities if (ent[1], ent[2]) in true_entities_boundary2type.keys()]
    type_error_entities = [ent for ent in boundary_correct_entities if
                           ent[0] != true_entities_boundary2type[(ent[1], ent[2])]]
    pred_error_entities = pred_entities - (true_entities & pred_entities)

    return round(len(type_error_entities) / len(boundary_correct_entities), 4), \
           round(len(type_error_entities) / len(pred_error_entities), 4)


def get_boundary_error_rate(true_list, pred_list):
    """

    :param true_list: List[List], each element is a labels list
    :param pred_list: List[List], each element is a labels list
    :return: boundary_FP_error_rate (O predicted as I-type), boundary_FN_error_rate(I-type predicted as O)
    """
    num_examples = len(true_list)
    if any(isinstance(s, list) for s in true_list):
        true_list = [item for sublist in true_list for item in sublist + ['O']]
    if any(isinstance(s, list) for s in pred_list):
        pred_list = [item for sublist in pred_list for item in sublist + ['O']]

    fp_tokens_num, fn_tokens_num = 0, 0
    for label1, label2 in zip(true_list, pred_list):
        if label1 != label2:
            if label1.upper() == 'O':
                fp_tokens_num += 1
            else:
                fn_tokens_num += 1
    origin_token_num = len(true_list) - num_examples
    return round(fp_tokens_num / origin_token_num, 4), round(fn_tokens_num / origin_token_num, 4)


def save_scores_as_png(score_list, output_dir, only_f1=False):
    """
        Save as bar picture.
    :param score_list: [(entity_length, (p, r, f1)), ...]
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prf1.png")
    cnt = 1
    while os.path.exists(output_path):
        cnt += 1
        output_path = os.path.join(output_dir, f"prf{cnt}.png")

    tick_label = [item[0] for item in score_list]
    x = np.arange(len(tick_label))

    if only_f1:
        bar_width = 0.6
        plt.figure(figsize=(20, 16), dpi=80)
        f1s = [round(item[1][-1] * 100, 2) for item in score_list]
        plt.bar(x, f1s, width=bar_width, color="blue", label="F1")
        for i in range(len(f1s)):
            plt.text(x[i], f1s[i] + 0.1, '%.02f' % f1s[i], ha='center', fontsize=10)
        plt.legend()
        plt.xticks(x, tick_label)
    else:
        bar_width = 0.3
        plt.figure(figsize=(50, 20), dpi=80)
        precisions = [round(item[1][0] * 100, 2) for item in score_list]
        recalls = [round(item[1][1] * 100, 2) for item in score_list]
        f1s = [round(item[1][2] * 100, 2) for item in score_list]
        plt.bar(x, precisions, width=bar_width, color="orchid", label="Precision")
        plt.bar(x + bar_width, recalls, width=bar_width, color="salmon", label="Recall")
        plt.bar(x + 2 * bar_width, f1s, width=bar_width, color="blue", label="F1")
        for i in range(len(f1s)):
            plt.text(x[i], precisions[i] + 0.05, '%.02f' % precisions[i], ha='center', fontsize=10)
            plt.text(x[i] + bar_width, recalls[i] + 0.1, '%.02f' % recalls[i], ha='center', fontsize=10)
            plt.text(x[i] + 2 * bar_width, f1s[i] + 0.05, '%.02f' % f1s[i], ha='center', fontsize=10)
        plt.legend()
        plt.xticks(x + bar_width, tick_label)

    plt.savefig(output_path)
    # plt.show()


if __name__ == '__main__':
    starts = [['O', 'O', 'O', 'MISC', 'O', 'O', 'O'], ['PER', 'O', 'O']]
    ends = [['O', 'O', 'O', 'O', 'O', 'MISC', 'O'], ['O', 'PER', 'O']]
    # print(convert_span_to_bio(starts, ends))

    # score_list = [(1, (0.8571428571428571, 0.8571428571428571, 0.8571428571428571)),
    #               (2, (0.9314285714285714, 0.6965811965811965, 0.7970660146699267)),
    #               (3, (0.9683257918552036, 0.8136882129277566, 0.8842975206611571)),
    #               (4, (0.9833333333333333, 0.8832335329341318, 0.9305993690851735)),
    #               (5, (0.9473684210526315, 0.8135593220338984, 0.8753799392097265)),
    #               (6, (0.9154929577464789, 0.7471264367816092, 0.8227848101265823)),
    #               (7, (0.95, 0.8382352941176471, 0.890625)),
    #               (8, (0.9315068493150684, 0.8607594936708861, 0.8947368421052632)),
    #               (9, (0.8947368421052632, 0.9444444444444444, 0.918918918918919)),
    #               (10, (0.9047619047619048, 0.926829268292683, 0.9156626506024096)),
    #               (11, (0.84375, 0.9310344827586207, 0.8852459016393444)),
    #               (12, (0.9285714285714286, 0.9811320754716981, 0.9541284403669724)),
    #               (13, (0.8333333333333334, 0.9615384615384616, 0.8928571428571429)),
    #               (14, (0.85, 0.8947368421052632, 0.8717948717948718)),
    #               (15, (0.9047619047619048, 0.95, 0.9268292682926829)),
    #               (16, (0.8947368421052632, 0.9444444444444444, 0.918918918918919)),
    #               (17, (0.8333333333333334, 1.0, 0.9090909090909091)),
    #               (18, (0.8, 0.5714285714285714, 0.6666666666666666)), (19, (1.0, 0.6, 0.7499999999999999)),
    #               (20, (1.0, 1.0, 1.0)), (21, (1.0, 1.0, 1.0)), (26, (0, 0.0, 0)), (27, (0, 0.0, 0)), (28, (0, 0.0, 0))]
    # save_scores_as_png(score_list, output_dir="./test_png", only_f1=True)
