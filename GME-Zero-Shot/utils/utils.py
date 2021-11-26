from collections import Counter
import re
from config import Config
config = Config()


def gpu_wrapper(item):
    if config.gpu:
        # print(item)
        return item.cuda()
    else:
        return item


def pretty_string(flt):
    ret = '%.6f' % flt
    if flt >= 0:
        ret = "+" + ret
    return ret


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess: str, answers: list):
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split())
        for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


def f1_score(predictions, references):
    assert len(predictions) == len(references)

    return 1.0 * sum(_f1_score(guess, answers) for guess, answers in zip(predictions, references)) / len(predictions)
