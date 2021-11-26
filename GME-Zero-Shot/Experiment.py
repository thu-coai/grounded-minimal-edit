import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import nltk
from torch.utils.data import TensorDataset
from dataloaders.blendedskilltalk import BlendedSkillTalk
from Modules.Losses.SeqLoss import SeqLoss
from tqdm import tqdm
import random
from config import Config
from utils.utils import gpu_wrapper, pretty_string, f1_score
from torch.utils.data import DataLoader
from Modules.transformers import (GPT2Tokenizer,
                                  PreTrainedSequenceLabeling,
                                  PreTrainedSeq2Seq,
                                  PreTrainedSeq2SeqBP,
                                  )
from bert_eval import BertEval
from utils.multi_bleu import calc_bleu_score
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

config = Config()
ROUND = config.ROUND
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if config.gpu:
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Experiment(object):

    def __init__(self):

        print('----- Loading tokenizer -----')
        print('[Reminder] Never use self.tokenizer.vocab_size!')
        if os.path.basename(config.model_name_or_path) == 'gpt2':
            tokenizer_class = GPT2Tokenizer
        else:
            raise ValueError()

        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
        )
        print('Before add special tokens: vocabulary size = {}'.format(len(self.tokenizer)))
        special_tokens_dict = {'eos_token': '<|speaker1|>',  # used as the token of speaker 1
                               'bos_token': '<|speaker2|>',  # used as the token of speaker 2
                               'cls_token': '<|cls|>',
                               'sep_token': '<|sep|>',
                               'pad_token': '<|pad|>',
                               }

        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print('We have added', num_added_toks, 'tokens')
        print('After add special tokens: vocabulary size = {}'.format(len(self.tokenizer)))

        print('----- Loading data -----')
        if config.model in ['blender-no-persona-cprm',
                            'blender-full-cprm',
                            'blender-full',
                            'blender-no-persona',
                            'transfertransfo',
                            ]:

            BST_DATASETS = {
                'blendedskilltalk': BlendedSkillTalk,
            }
            bst_dataset_class = BST_DATASETS[config.task]
            # self.bst_val_set = bst_dataset_class('valid', self.tokenizer, config.model)
            self.bst_test_set = bst_dataset_class('test', self.tokenizer, config.model)
            # print('The bst val set has {} items'.format(len(self.bst_val_set)))
            print('The bst test set has {} items'.format(len(self.bst_test_set)))
        elif config.model in ['persona-labeling',
                              ]:
            pass
        else:
            raise ValueError()

        print('----- Loading model -----')
        criterionSeq = SeqLoss(voc_size=len(self.tokenizer),
                               pad=self.tokenizer.pad_token_id,
                               bos=self.tokenizer.bos_token_id,
                               eos=self.tokenizer.eos_token_id,
                               unk=self.tokenizer.unk_token_id,
                               prob=False)
        if config.model in ['blender-no-persona-cprm',
                            'blender-full-cprm',
                            'transfertransfo',
                            ]:
            self.model = PreTrainedSeq2Seq.from_pretrained(pretrained_model_name_or_path=config.model_name_or_path,
                                                           layers=config.layers,
                                                           tokenizer=self.tokenizer,
                                                           decoder_max_len=config.max_response_len,
                                                           criterionSeq=criterionSeq)
        elif config.model in ['blender-full',
                              'blender-no-persona',
                              ]:
            pass
        elif config.model in ['persona-labeling',
                              ]:
            if config.task == 'blendedskilltalk':
                label_weight = self.get_label_weight(occur=[1465882, 43581])
            else:
                raise ValueError()
            self.model = PreTrainedSequenceLabeling.from_pretrained(pretrained_model_name_or_path=config.model_name_or_path,
                                                                    layers='default',
                                                                    tokenizer=self.tokenizer,
                                                                    n_label=label_weight.shape[0],
                                                                    label_weight=label_weight,
                                                                    criterionSeq=criterionSeq)
        else:
            raise ValueError()

        if config.model in ['blender-no-persona-cprm',
                            'blender-full-cprm',
                            'persona-labeling',
                            'transfertransfo',
                            ]:
            self.modules = ['model']
        elif config.model in ['blender-full',
                              'blender-no-persona',
                              ]:
            self.modules = []
        else:
            raise ValueError()
        for module in self.modules:
            print('--- {}: '.format(module))
            print(getattr(self, module))
            if getattr(self, module) is not None:
                setattr(self, module, gpu_wrapper(getattr(self, module)))
                if config.gpu and len(config.gpu_ids) > 1:
                    print('Enabling multi-gpu training...')
                    setattr(self, 'Parallel_' + module, nn.DataParallel(getattr(self, module), device_ids=config.gpu_ids))

        self.scopes = {'gen': ['model']}
        for scope in self.scopes.keys():
            setattr(self, scope + '_lr', getattr(config, scope + '_lr'))

        self.iter_num = 0
        self.logger = None
        if config.model in ['blender-no-persona-cprm',
                            'blender-full-cprm',
                            'blender-full',
                            'blender-no-persona',
                            'transfertransfo',
                            'persona-labeling',
                            ]:
            self.best_metric = - float('inf')
        else:
            raise ValueError()
        self.decay_num = 0
        self.no_improvement = 0

    def get_label_weight(self, occur):
        occur = torch.FloatTensor(occur)
        freq = occur / occur.sum()
        label_weight = 1 / freq / freq.shape[0]
        return label_weight

    def restore_from(self, module, path, strict=True):
        print('Loading the trained best models from {}...'.format(path))
        path = os.path.join(path, 'best-{}.ckpt'.format(module))
        getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=strict)

    def restore_model(self, modules, dirs=None):
        print('Loading the trained best models...')
        if dirs is not None:
            assert len(modules) == len(dirs)
            for module, directory in zip(modules, dirs):
                path = os.path.join(directory, 'best-{}.ckpt'.format(module))
                getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),
                                                      strict=True)
        else:
            for module in modules:
                path = os.path.join(config.save_model_dir, 'best-{}.ckpt'.format(module))
                getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),
                                                      strict=True)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from utils.logger import Logger
        self.logger = Logger(config.log_dir)

    def log_step(self, loss):
        # Log loss.
        for loss_name, value in loss.items():
            self.logger.scalar_summary(loss_name, value, self.iter_num)
        # Log learning rate.
        for scope in self.scopes:
            self.logger.scalar_summary('{}/lr'.format(scope), getattr(self, scope + '_lr'), self.iter_num)

    def save_step(self, modules, use_iter=False):
        save_dir = config.save_model_dir
        if use_iter:
            for module in modules:
                path = os.path.join(save_dir, '{}-{}.ckpt'.format(self.iter_num, module))
                torch.save(getattr(self, module).state_dict(), path)
        else:
            for module in modules:
                path = os.path.join(save_dir, 'best-{}.ckpt'.format(module))
                torch.save(getattr(self, module).state_dict(), path)
        print('Saved model checkpoints into {}...\n\n\n\n\n\n\n\n\n\n\n\n'.format(save_dir))

    def zero_grad(self):
        for scope in self.scopes:
            getattr(self, scope + '_optim').zero_grad()

    def step(self, scopes):
        # Gather parameters.
        grouped_params = []
        for scope in scopes:
            grouped_params.extend(getattr(self, scope + '_grouped_parameters'))

        # Clip on all parameters.
        if config.fp16:
            raise NotImplementedError()
        else:
            torch.nn.utils.clip_grad_norm_(grouped_params, config.max_grad_norm)

        for scope in scopes:
            # Optimize.
            getattr(self, scope + '_optim').step()
            # Schedule.
            # getattr(self, scope + '_scheduler').step()

    def update_lr_by_half(self):
        self.decay_num += 1
        for scope in self.scopes:
            setattr(self, scope + '_lr', getattr(self, scope + '_lr') / 2)  # Half the learning rate.
            for param_group in getattr(self, scope + '_optim').param_groups:
                param_group['lr'] = getattr(self, scope + '_lr')
            print('{}: {}'.format(scope + '_lr', getattr(self, scope + '_lr')))

    def set_requires_grad(self, modules, requires_grad):
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            for param in getattr(self, module).parameters():
                param.requires_grad = requires_grad

    def set_training(self, mode):
        for module in self.modules:
            if getattr(self, module) is not None:
                getattr(self, module).train(mode=mode)

    def test(self):

        # Evaluate.
        if config.model in ['blender-no-persona-cprm',
                            'blender-full-cprm',
                            'transfertransfo',
                            ]:
            self.restore_model(['model'])
        elif config.model in ['blender-full',
                              'blender-no-persona',
                              ]:
            pass
        else:
            raise ValueError()

        return self.bst_evaluate_multiref(test=True)

    def persona_labeling(self, responses, response_lens):
        self.set_training(mode=False)

        # Pretrained model.
        self.restore_model(['model'])

        the_set = TensorDataset(responses, response_lens)
        eval_dataloader = DataLoader(the_set, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers)

        # Eval!
        print("\n\n\n\n***** Running evaluation *****")
        print("  Num examples = {}".format(len(the_set)))
        print("  Batch size = {}".format(config.eval_batch_size))

        batchify = {
            'response': [],
            'response_len': [],
            'persona_probs': [],
        }

        top_5 = [False, True][0]
        tot = 0
        with torch.no_grad():
            if config.display_tqdm:
                pbar = tqdm(eval_dataloader, desc="Evaluating")
            else:
                pbar = eval_dataloader
            for data in pbar:
                tot += 1
                if tot > 5 and top_5:
                    break
                data = self.cuda_data(data)
                t_cls_sep, t_len = data

                if config.bucket:
                    t_cls_sep = self.bucketize(t_cls_sep, t_len)

                batch_size = t_cls_sep.shape[0]

                persona_probs = self.multi_gpu_forward(encoder_input_ids=t_cls_sep,
                                                       encoder_labels=None,
                                                       mode='persona_prob',
                                                       module='model',
                                                       batch_size=batch_size)

                batchify['persona_probs'].append(persona_probs)

                response = self.strip_cls_sep([[self.tokenizer.convert_ids_to_tokens(idx.item()) for idx in sent] for sent in t_cls_sep])
                batchify['response'].append(response)
                batchify['response_len'].append(t_len)

        # ----- De-batchify -----
        for key in batchify.keys():
            if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
            elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                temp = []
                for batch in batchify[key]:
                    temp.extend(batch)
                batchify[key] = temp

        original_responses = []
        mask_probs = []
        for response, response_len, persona_probs in zip(batchify['response'], batchify['response_len'], batchify['persona_probs']):
            mask_prob = []
            assert len(response) == response_len.item()
            for token, persona_prob in zip(response, persona_probs[1:response_len.item() + 1]):
                mask_prob.append(persona_prob.item())
            original_responses.append(response)
            mask_probs.append(mask_prob)

        self.set_training(mode=True)

        return original_responses, mask_probs

    def bst_evaluate_multiref(self, test=False):
        self.set_training(mode=False)

        if test:
            the_set = self.bst_test_set
        else:
            the_set = self.bst_val_set

        eval_dataloader = DataLoader(the_set, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers)

        # Eval!
        print("\n\n\n\n***** Running evaluation *****")
        print("  Num examples = {}".format(len(the_set)))
        print("  Batch size = {}".format(config.eval_batch_size))

        assert len(the_set[0]) % 2 != 0
        n_ref = (len(the_set[0]) - 5) // 2
        batchify = {
            'preds': [],
            'context': [],
            'original_response': [],
            'original_response_len': [],
        }
        for i in range(n_ref):
            batchify['response{}'.format(i)] = []
            batchify['response_len{}'.format(i)] = []

        if config.display_tqdm:
            pbar = tqdm(eval_dataloader, desc="Evaluating")
        else:
            pbar = eval_dataloader
        for data in pbar:
            data = self.cuda_data(data)
            cls_sep, s_len, type_ids, or_cls_sep, or_len = data[:5]
            batch_size = cls_sep.shape[0]

            if config.bucket:
                cls_sep = self.bucketize(cls_sep, s_len)
                type_ids = self.bucketize(type_ids, s_len)
                # Do not bucketize or_cls_sep (for backprop)

            if config.model in ['blender-no-persona-cprm',
                                'blender-full-cprm',
                                'transfertransfo',
                                ]:
                with torch.no_grad():
                    preds = self.multi_gpu_forward(input_ids=cls_sep,
                                                   type_ids=type_ids,
                                                   s_len=s_len,
                                                   mode='forward',
                                                   module='model',
                                                   batch_size=batch_size)
            elif config.model in ['blender-full',
                                  'blender-no-persona',
                                  ]:
                preds = or_cls_sep[:, 1:]  # Copy the blender outputs.
            else:
                raise ValueError()

            context = self.strip_cls_sep([[self.tokenizer.convert_ids_to_tokens(idx.item()) for idx in sent] for sent in cls_sep])
            batchify['context'].append(context)

            original_response = self.strip_cls_sep([[self.tokenizer.convert_ids_to_tokens(idx.item()) for idx in sent] for sent in or_cls_sep])
            batchify['original_response'].append(original_response)

            for i in range(n_ref):
                ri_cls_sep = data[5 + 2 * i]
                ri_len = data[5 + 2 * i + 1]
                response_i = self.strip_cls_sep([[self.tokenizer.convert_ids_to_tokens(idx.item()) for idx in sent] for sent in ri_cls_sep])
                batchify['response{}'.format(i)].append(response_i)
                batchify['response_len{}'.format(i)].append(ri_len)

            preds = self.strip_sep([[self.tokenizer.convert_ids_to_tokens(idx.item()) for idx in pred] for pred in preds])
            batchify['preds'].append(preds)

        # ----- De-batchify -----
        for key in batchify.keys():
            if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
            elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                temp = []
                for batch in batchify[key]:
                    temp.extend(batch)
                batchify[key] = temp

        Ref_bleu_moses = calc_bleu_score(
            predictions=[' '.join(nltk.tokenize.WordPunctTokenizer().tokenize(self.tokenizer.convert_tokens_to_string(sent))) for sent in batchify['preds']],
            references=[[' '.join(nltk.tokenize.WordPunctTokenizer().tokenize(self.tokenizer.convert_tokens_to_string(senti))) for senti in sents]
                        for sents in
                        zip(*(batchify['response{}'.format(i)] for i in range(n_ref)))],
            log_dir=config.tmp_dir,
            multi_ref=True
        )
        print('\n\n\nMoses reference BLEU:', Ref_bleu_moses)

        f1 = f1_score(
            predictions=[' '.join(nltk.tokenize.WordPunctTokenizer().tokenize(self.tokenizer.convert_tokens_to_string(sent))) for sent in batchify['preds']],
            references=[[' '.join(nltk.tokenize.WordPunctTokenizer().tokenize(self.tokenizer.convert_tokens_to_string(senti))) for senti in sents]
                        for sents in zip(*(batchify['response{}'.format(i)] for i in range(n_ref)))],
        )
        print('f1 score:', f1)

        if hasattr(self, 'cls_evaluator'):
            pass
        else:
            setattr(self, 'cls_evaluator', BertEval(load_dataset=False))
        score = getattr(self, 'cls_evaluator').class_score(preds=[self.tokenizer.convert_tokens_to_string(sent) for sent in batchify['preds']],
                                                           data_mode='test' if test else 'valid',
                                                           detail=False)
        print('P score:', score)

        mean = (Ref_bleu_moses + score) / 2

        metric = mean
        no_improvement = True
        if not test and metric > self.best_metric:
            print('Best metric found.')
            no_improvement = False
            self.best_metric = metric
            self.save_step(['model'])

        self.set_training(mode=True)

        if test:
            predictions = [' '.join(nltk.tokenize.WordPunctTokenizer().tokenize(self.tokenizer.convert_tokens_to_string(sent))) for sent in batchify['preds']]
            return predictions
        else:
            return no_improvement

    def number_parameters(self):
        print('Number of parameters', sum(p.numel() for p in self.model.parameters()))

    def strip_cls_sep(self, sents):
        if isinstance(sents[0], list):
            return [sent[1:max(sent.index(self.tokenizer.sep_token), 1)] if self.tokenizer.sep_token in sent else sent[1:]
                    for sent in sents]
        else:
            sent = sents
            return sent[1:max(sent.index(self.tokenizer.sep_token), 1)] if self.tokenizer.sep_token in sent else sent[1:]

    def strip_sep(self, sents):
        if isinstance(sents[0], list):
            return [sent[:max(sent.index(self.tokenizer.sep_token), 1)] if self.tokenizer.sep_token in sent else sent
                    for sent in sents]
        else:
            sent = sents
            return sent[:max(sent.index(self.tokenizer.sep_token), 1)] if self.tokenizer.sep_token in sent else sent

    @staticmethod
    def cuda_data(data):
        return [gpu_wrapper(item) for item in data]

    @staticmethod
    def bucketize(cls_sep, seq_len):
        max_len = max(seq_len).item()
        cls_sep = cls_sep[:, :max_len + 2].contiguous()
        return cls_sep

    def multi_gpu_forward(self, *positional_args, **kwargs):
        """Data parallel handled it well. """
        module = kwargs.pop('module')
        batch_size = kwargs.pop('batch_size')

        if hasattr(self, 'Parallel_{}'.format(module)):
            return getattr(self, 'Parallel_{}'.format(module))(*positional_args, **kwargs)
        else:
            return getattr(self, module)(*positional_args, **kwargs)
