import os
import shutil


def calc_bleu_score(predictions, references, log_dir=None, multi_ref=False):
    shutil.rmtree(log_dir)
    os.mkdir(log_dir)

    pred_file = os.path.join(log_dir, 'pred.txt')
    ref_file = os.path.join(log_dir, 'ref.txt')

    if multi_ref:
        num_sents = len(references)
        num_refs = len(references[0])
        # max_num_refs = max([len(reference) for reference in references])
        for ref_idx in range(num_refs):
        # for ref_idx in range(max_num_refs):
            with open(ref_file + str(ref_idx), 'w', encoding='utf-8') as f:
                for sent_idx in range(num_sents):
                    # if len(references[sent_idx]) > ref_idx:
                    #     print(references[sent_idx][ref_idx], file=f, )
                    # else:
                    #     print('', file=f, )
                    print(references[sent_idx][ref_idx], file=f, )
    else:
        with open(ref_file, 'w', encoding='utf-8') as f:
            for s in references:
                print(s, file=f, )
    with open(pred_file, 'w', encoding='utf-8') as f:
        for s in predictions:
            print(s, file=f, )

    temp = os.path.join(log_dir, "result.txt")

    command = "perl utils/multi-bleu.perl " + ref_file + "<" + pred_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    print(result)
    penalty = float(result.split()[4][4:-1])
    bleu1 = float(result.split()[3].split('/')[0])
    bleu2 = float(result.split()[3].split('/')[1])
    bleu3 = float(result.split()[3].split('/')[2])
    bleu4 = float(result.split()[3].split('/')[3])
    print('BLEU1 / BLEU2 / BLEU3 / BLEU4 = {} / {} / {} / {}'.format(round(bleu1, 1), round(bleu2, 1), round(bleu3, 1), round(bleu4, 1)))
    score = float(result.split()[2][:-1])

    return score
