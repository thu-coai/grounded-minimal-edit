import torch

import os
import numpy as np
from tqdm import tqdm
from utils.utils import gpu_wrapper
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from dataloaders.dnli import DNLI, PersonanliProcessor, label_convert, convert_examples_to_features
import random
import json
from functools import reduce


class TempData(Dataset):
    """The DNLI-style temp dataset."""

    def __init__(self, turn, persona_item, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.processor = PersonanliProcessor()
        self.label_list = self.processor.get_labels()

        examples = self.processor.create_batch(turn, persona_item, labels)
        self.features = convert_examples_to_features(examples, self.label_list, self.max_len, self.tokenizer)

        self.voc_size = len(self.tokenizer)

        print('Temp data successfully read.')

    def __getitem__(self, index):
        f = self.features[index]
        input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        input_mask = torch.tensor(f.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(f.segment_ids, dtype=torch.long)
        label_ids = torch.tensor([f.label_id], dtype=torch.long).squeeze(0)

        return input_ids, input_mask, segment_ids, label_ids

    def __len__(self):
        return len(self.features)


class BertEval(object):

    def __init__(self, load_dataset):

        self.model_name_or_path = 'bert-base-uncased'
        self.max_seq_length = 128
        self.do_lower_case = True
        self.train_batch_size = 32
        self.eval_batch_size = 32

        self.use_local = True
        if self.use_local:
            DIR = '../pretrained_transformer_weights'
            self.model_name_or_path = os.path.join(DIR, self.model_name_or_path)

        print('----- Loading tokenizer -----')
        print('[Reminder] Never use self.tokenizer.vocab_size!')
        tokenizer_class = BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_name_or_path,
            do_lower_case=self.do_lower_case,
        )
        print('self.tokenizer.pad_token', self.tokenizer.pad_token)
        print('self.tokenizer.pad_token_id', self.tokenizer.pad_token_id)
        print('self.tokenizer.unk_token', self.tokenizer.unk_token)
        print('self.tokenizer.unk_token_id', self.tokenizer.unk_token_id)
        print('self.tokenizer.cls_token', self.tokenizer.cls_token)
        print('self.tokenizer.cls_token_id', self.tokenizer.cls_token_id)
        print('self.tokenizer.sep_token', self.tokenizer.sep_token)
        print('self.tokenizer.sep_token_id', self.tokenizer.sep_token_id)

        if load_dataset:
            self.test_set = DNLI('test', self.tokenizer, max_len=self.max_seq_length)
            self.verified_test_set = DNLI('verified_test', self.tokenizer, max_len=self.max_seq_length)
            print("len(self.test_set) = {}".format(len(self.test_set)))
            print("len(self.verified_test_set) = {}".format(len(self.verified_test_set)))

        print('----- Loading model -----')
        self.model = BertForSequenceClassification.from_pretrained(self.model_name_or_path, num_labels=3)
        assert self.model.config.num_labels == 3

        self.modules = ['model']
        for module in self.modules:
            print('--- {}: '.format(module))
            print(getattr(self, module))
            if getattr(self, module) is not None:
                setattr(self, module, gpu_wrapper(getattr(self, module)))

        self.in_before = False

    def restore_model(self):
        print('Loading the trained best models...')
        path = os.path.join('../dnli-bert/dnli_model.bin')
        getattr(self, 'model').load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),
                                               strict=True)

    def set_training(self, mode):
        for module in self.modules:
            if getattr(self, module) is not None:
                getattr(self, module).train(mode=mode)

    def test(self):
        self.restore_model()

        # Evaluate.
        print('Test.\n')
        self.evaluate(mode='test')

        print('Verified test.\n')
        self.evaluate(mode='verified_test')

    def evaluate(self, mode):
        self.set_training(mode=False)
        # Loop to handle MNLI double evaluation (matched, mis-matched)

        the_set = getattr(self, mode + '_set')

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(the_set)
        eval_dataloader = DataLoader(the_set, sampler=eval_sampler, batch_size=self.eval_batch_size, num_workers=0)

        # Eval!
        print("\n\n\n\n***** Running evaluation *****")
        print("  Num examples = {}".format(len(the_set)))
        print("  Batch size = {}".format(self.eval_batch_size))

        batchify = {
            'turn': [],
            'persona': [],
            'label': [],
            'pred': [],
        }

        top_five = [False, True][0]
        count = 0
        with torch.no_grad():
            for data in tqdm(eval_dataloader, desc="Evaluating"):
                count += 1
                if top_five and count > 10:
                    break
                data = self.cuda_data(data)
                input_ids, input_mask, segment_ids, label_ids = data
                batchify['label'].append(label_ids)

                logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
                preds = torch.argmax(logits, dim=1)
                batchify['pred'].append(preds)

                turn, persona = self.split_turn_persona([[self.tokenizer.convert_ids_to_tokens(idx.item()) for idx in sent] for sent in input_ids])
                batchify['turn'].append(turn)
                batchify['persona'].append(persona)

        # ----- De-batchify -----
        for key in batchify.keys():
            if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
            elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                temp = []
                for batch in batchify[key]:
                    temp.extend(batch)
                batchify[key] = temp

        acc = (batchify['label'] == batchify['pred']).float().sum().item() / batchify['label'].shape[0]

        print('\n\n\nacc:{}\n\n\n'.format(acc))

        confusion_matrix = np.array([[0, 0, 0] for _ in range(3)])
        for _pred, _label in zip(batchify['pred'], batchify['label']):
            confusion_matrix[_label.item(), _pred.item()] += 1
        print(confusion_matrix)

        peep_num = 0
        for sent_id in range(peep_num):
            print('turn:    {}'.format(' '.join(batchify['turn'][sent_id])))
            print('persona: {}'.format(' '.join(batchify['persona'][sent_id])))
            print('label:   {}'.format(batchify['label'][sent_id].item()))
            print('pred:    {}'.format(batchify['pred'][sent_id].item()))
            print()
            print('-' * 50)

        self.set_training(mode=True)

    def split_turn_persona(self, sents):
        if isinstance(sents[0], list):
            turns = [sent[1:max(sent.index(self.tokenizer.sep_token), 1)] for sent in sents]
            sents = [sent[max(sent.index(self.tokenizer.sep_token), 1) + 1:] for sent in sents]
            personas = [sent[:max(sent.index(self.tokenizer.sep_token), 1)] for sent in sents]
            return turns, personas
        else:
            sent = sents
            turn = sent[1:max(sent.index(self.tokenizer.sep_token), 1)]
            sent = sent[max(sent.index(self.tokenizer.sep_token), 1) + 1:]
            persona = sent[:max(sent.index(self.tokenizer.sep_token), 1)]
            return turn, persona

    def number_parameters(self):
        print('Number of parameters', sum(p.numel() for p in self.model.parameters()))

    def read_intervening_persona(self, data_mode):
        file_name = os.path.join('../data', 'personachat-ucpt', '{}.json'.format(data_mode))
        print(("Reading lines from {}".format(file_name)))

        intervening_persona = []
        with open(file_name) as f:
            data = json.load(f)
            for _data in data:
                intervening_persona.append(_data['intervening_persona'])

        return intervening_persona

    def class_score(self, preds, data_mode, detail=False):
        """

        :param preds: [str x N]
        :return: float, accuracy of classification.
        """
        self.set_training(mode=False)

        # Restore the pretrained model.
        if self.in_before:
            pass
        else:
            self.restore_model()
            self.in_before = True

        # Prepare data.
        intervening_persona = self.read_intervening_persona(data_mode)

        assert len(preds) == len(intervening_persona)
        turn = []
        persona_item = []
        labels = []
        for pred, _intervening_persona in zip(preds, intervening_persona):
            for __intervening_persona in _intervening_persona:
                turn.append(pred)
                persona_item.append(__intervening_persona)
                labels.append('entailment')  # Not used anyway.
        score_set = TempData(turn=turn, persona_item=persona_item, labels=labels, tokenizer=self.tokenizer, max_len=self.max_seq_length)

        # Note that DistributedSampler samples randomly
        dataloader = DataLoader(score_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0)

        batchify = {
            'turn': [],
            'persona': [],
            'pred_label': [],
        }

        with torch.no_grad():
            for data in dataloader:
                data = self.cuda_data(data)
                input_ids, input_mask, segment_ids, label_ids = data

                logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
                pred_labels = torch.argmax(logits, dim=1)
                batchify['pred_label'].append([score_set.label_list[pred_label.item()] for pred_label in pred_labels])

                turn, persona = self.split_turn_persona([[self.tokenizer.convert_ids_to_tokens(idx.item()) for idx in sent] for sent in input_ids])
                batchify['turn'].append(turn)
                batchify['persona'].append(persona)

        # ----- De-batchify -----
        for key in batchify.keys():
            if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
            elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                temp = []
                for batch in batchify[key]:
                    temp.extend(batch)
                batchify[key] = temp

        assert len(batchify['persona']) == len(batchify['pred_label']) == len(batchify['turn'])
        n_batch = len(preds)

        if detail:
            score = {'contradiction': 0,
                     'entailment': 0,
                     'neutral': 0,
                     }
            for pred_label in batchify['pred_label']:
                score[pred_label] += 1
            score['total'] = n_batch
        else:
            score_sum = 0
            for pred_label in batchify['pred_label']:
                if pred_label == 'contradiction':
                    score_sum -= 0.5
                elif pred_label == 'entailment':
                    score_sum += 0.5
                elif pred_label == 'neutral':
                    pass
                else:
                    raise ValueError()
            score = 100.0 * score_sum / n_batch

        self.set_training(mode=True)

        return score

    def strip_cls_sep(self, sents):
        if isinstance(sents[0], list):
            return [sent[1:max(sent.index(self.tokenizer.sep_token), 1)] if self.tokenizer.sep_token in sent else sent
                    for sent in sents]
        else:
            sent = sents
            return sent[1:max(sent.index(self.tokenizer.sep_token), 1)] if self.tokenizer.sep_token in sent else sent

    def strip_sep(self, sents):
        if isinstance(sents[0], list):
            return [sent[:max(sent.index(self.tokenizer.sep_token), 1)] if self.tokenizer.sep_token in sent else sent
                    for sent in sents]
        else:
            sent = sents
            return sent[:max(sent.index(self.tokenizer.sep_token), 1)] if self.tokenizer.sep_token in sent else sent

    def strip_pad(self, sents):
        if isinstance(sents[0], list):
            return [sent[:max(sent.index(self.tokenizer.pad_token), 1)] if self.tokenizer.pad_token in sent else sent
                    for sent in sents]
        else:
            sent = sents
            return sent[:max(sent.index(self.tokenizer.pad_token), 1)] if self.tokenizer.pad_token in sent else sent

    @staticmethod
    def cuda_data(data):
        return [gpu_wrapper(item) for item in data]


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    mode = ['test'][0]
    if mode == 'test':
        ClsEval = BertEval(load_dataset=True)
        ClsEval.test()
    else:
        raise ValueError()
