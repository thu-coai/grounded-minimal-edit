import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import nltk
from torch.utils.data import TensorDataset
from dataloaders.personachat_ucpt import PersonaChat, PersonaChatUCPT, PersonaChatPersonaLabeling
from Modules.Losses.SeqLoss import SeqLoss
from tqdm import tqdm
import random
from config import Config
from utils.utils import gpu_wrapper, pretty_string
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
        if os.path.basename(config.model_name_or_path) in ['gpt2',
                                                           'DialoGPT-small',
                                                           ]:
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
        if config.model in ['cprm',
                            'transfertransfo',
                            'backprop',
                            ]:
            if config.model != 'backprop':
                DIALOGUE_DATASETS = {
                    'persona-chat': PersonaChat,
                }
                dialogue_dataset_class = DIALOGUE_DATASETS[config.task]
                self.dialogue_train_set = dialogue_dataset_class('train', self.tokenizer, config.model)
                print('The dialogue train set has {} items'.format(len(self.dialogue_train_set)))

            UCPT_DATASETS = {
                'persona-chat': PersonaChatUCPT,
            }
            ucpt_dataset_class = UCPT_DATASETS[config.task]
            self.ucpt_val_set = ucpt_dataset_class('valid', self.tokenizer, config.model)
            self.ucpt_test_set = ucpt_dataset_class('test', self.tokenizer, config.model)
            print('The ucpt val set has {} items'.format(len(self.ucpt_val_set)))
            print('The ucpt test set has {} items'.format(len(self.ucpt_test_set)))
        elif config.model in ['persona-labeling',
                              ]:
            PERSONA_LABELING_DATASETS = {
                'persona-chat': PersonaChatPersonaLabeling,
            }
            persona_labeling_dataset_class = PERSONA_LABELING_DATASETS[config.task]
            self.persona_labeling_train_set = persona_labeling_dataset_class('train', self.tokenizer, config.model)
            print('The persona labeling train set has {} items'.format(len(self.persona_labeling_train_set)))
        else:
            raise ValueError()

        print('----- Loading model -----')
        criterionSeq = SeqLoss(voc_size=len(self.tokenizer),
                               pad=self.tokenizer.pad_token_id,
                               bos=self.tokenizer.bos_token_id,
                               eos=self.tokenizer.eos_token_id,
                               unk=self.tokenizer.unk_token_id,
                               prob=False)
        if config.model in ['transfertransfo',
                            ]:
            self.model = PreTrainedSeq2Seq.from_pretrained(pretrained_model_name_or_path=config.model_name_or_path,
                                                           layers=config.layers,
                                                           tokenizer=self.tokenizer,
                                                           decoder_max_len=config.max_response_len,
                                                           criterionSeq=criterionSeq)
        elif config.model == 'cprm':
            self.model = PreTrainedSeq2Seq.from_pretrained(pretrained_model_name_or_path=config.model_name_or_path,
                                                           layers=config.layers,
                                                           smooth_eps=config.smooth_eps,
                                                           tokenizer=self.tokenizer,
                                                           decoder_max_len=config.max_response_len,
                                                           criterionSeq=criterionSeq)
        elif config.model == 'backprop':
            self.model = PreTrainedSeq2SeqBP.from_pretrained(pretrained_model_name_or_path=config.model_name_or_path,
                                                             layers=config.layers,
                                                             tokenizer=self.tokenizer,
                                                             decoder_max_len=config.max_response_len,
                                                             criterionSeq=criterionSeq,
                                                             num_iter=config.num_iter,
                                                             fgm_iters=config.fgm_iters,
                                                             stepsize=config.stepsize,
                                                             mix_rate=config.mix_rate)
        elif config.model in ['persona-labeling',
                              ]:
            if config.task == 'persona-chat':
                if config.model == 'persona-labeling':
                    label_weight = self.get_label_weight(occur=config.label_occurance)  # float('inf')

                else:
                    raise ValueError()
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

        if config.model in ['cprm',
                            'transfertransfo',
                            'backprop',
                            'persona-labeling',
                            ]:
            self.modules = ['model']
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
        if config.model in ['cprm',
                            'transfertransfo',
                            'backprop',
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

    def train(self):

        if config.model == 'backprop':  # Validation only.
            raise NotImplementedError()

        # Logging.
        if config.use_tensorboard:
            self.build_tensorboard()

        self.build_optim()

        if config.model in ['cprm',
                            'transfertransfo',
                            ]:
            train_dataset = self.dialogue_train_set
        elif config.model in ['persona-labeling',
                              ]:
            train_dataset = self.persona_labeling_train_set
        else:
            raise ValueError()

        # Train!
        print("***** Running training *****")
        print("  Training num examples = {}".format(len(train_dataset)))
        print("  Instantaneous batch size per GPU = {}".format(1.0 * config.train_batch_size / len(config.gpu_ids) / config.gradient_accumulation_steps))
        print("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(config.train_batch_size))
        print("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)

        # Train.
        epoch = 0
        self.zero_grad()
        try:
            while True:
                self.train_epoch(train_dataset, epoch)
                epoch += 1
                if self.decay_num >= config.max_decay_num:
                    break
        except KeyboardInterrupt:
            print('-' * 100)
            print('Training not finished. {} steps finished. Quit training.'.format(self.iter_num))

        # Test.
        self.test()

    def build_optim(self):
        # Set trainable parameters, according to the frozen parameter list.
        for scope in self.scopes.keys():
            optimizer_grouped_parameters = [
                {'params': [],
                 'weight_decay': config.weight_decay},
                {'params': [],
                 'weight_decay': 0.0},
                {'params': [],
                 'weight_decay': 'cannot be optimized'},
            ]
            no_decay = ['bias', 'LayerNorm.weight']

            for module in self.scopes[scope]:
                if getattr(self, module) is not None:
                    for n, p in getattr(self, module).named_parameters():
                        # k is the parameter name; v is the parameter value.
                        if p.requires_grad:
                            print("[{} Trainable:]".format(module), n)
                            # Weight decay.
                            if not any(nd in n for nd in no_decay):
                                optimizer_grouped_parameters[0]['params'].append(p)
                            else:
                                optimizer_grouped_parameters[1]['params'].append(p)
                        else:
                            print("[{} Frozen:]".format(module), n)
                            # Frozen.
                            optimizer_grouped_parameters[2]['params'].append(p)

            if config.optimizer == 'sgd':
                setattr(self, scope + '_optim', SGD(optimizer_grouped_parameters[0]['params'] + optimizer_grouped_parameters[1]['params'],
                                                    lr=getattr(self, scope + '_lr'),
                                                    ))
            elif config.optimizer == 'adam':
                setattr(self, scope + '_optim', Adam(optimizer_grouped_parameters[0]['params'] + optimizer_grouped_parameters[1]['params'],
                                                     lr=getattr(self, scope + '_lr'),
                                                     eps=config.adam_epsilon))
            else:
                raise ValueError()

            setattr(self,
                    scope + '_grouped_parameters',
                    optimizer_grouped_parameters[0]['params'] + optimizer_grouped_parameters[1]['params'])

            if config.fp16:
                raise NotImplementedError()

    def test(self):
        if config.model != 'backprop':
            self.restore_model(['model'])
        else:
            name_components = [config.task,
                               'transfertransfo',
                               '{}'.format(config.seed),
                               ]
            restore_path = os.path.join('outputs', 'saved_model', '-'.join(name_components), 'best-model.ckpt')
            self.model.load_state_dict(torch.load(restore_path, map_location=lambda storage, loc: storage),
                                       strict=True)

        # Evaluate.
        if config.model in ['cprm',
                            'transfertransfo',
                            'backprop',
                            ]:
            return self.ucpt_evaluate_multiref(test=True)
        elif config.model in ['persona-labeling',
                              ]:
            pass
        else:
            raise ValueError()

    def train_epoch(self, train_dataset, epoch_id):

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config.train_batch_size // config.gradient_accumulation_steps,
                                      shuffle=True,
                                      num_workers=config.num_workers)

        if config.display_tqdm:
            pbar = tqdm(train_dataloader, desc="Iteration", disable=False)
        else:
            pbar = train_dataloader
        for step, data in enumerate(pbar):
            self.iter_num += 1
            self.set_training(mode=True)
            loss = {}
            data = self.cuda_data(data)

            if config.model in ['cprm',
                                'transfertransfo',
                                ]:
                cls_sep, s_len, t_len, type_ids = data

                if config.bucket:
                    cls_sep = self.bucketize(cls_sep, s_len + t_len)
                    type_ids = self.bucketize(type_ids, s_len + t_len)

                batch_size = cls_sep.shape[0]
            elif config.model in ['persona-labeling',
                                  ]:
                t_cls_sep, t_len, p_label = data

                if config.bucket:
                    t_cls_sep = self.bucketize(t_cls_sep, t_len)

                batch_size = t_cls_sep.shape[0]
            else:
                raise ValueError()

            if config.model in ['cprm',
                                'transfertransfo',
                                ]:
                seq_loss = self.multi_gpu_forward(input_ids=cls_sep,
                                                  type_ids=type_ids,
                                                  s_len=s_len,
                                                  mode='forward',
                                                  module='model',
                                                  batch_size=batch_size)

                # Data parallel.
                seq_loss = seq_loss.mean(0)

                tot_loss = seq_loss

                # ----- Logging -----
                loss['seq/L'] = round(seq_loss.item(), ROUND)

                if config.gradient_accumulation_steps > 1:
                    tot_loss = tot_loss / config.gradient_accumulation_steps

                if config.fp16:
                    raise NotImplementedError()
                else:
                    tot_loss.backward()

                if self.iter_num % config.gradient_accumulation_steps == 0:
                    # ----- Backward for scopes: ['gen'] -----
                    self.step(['gen'])
                    self.zero_grad()

            elif config.model in ['persona-labeling',
                                  ]:
                labeling_loss = self.multi_gpu_forward(encoder_input_ids=t_cls_sep,
                                                       encoder_labels=p_label,
                                                       mode='forward',
                                                       module='model',
                                                       batch_size=batch_size)

                # Data parallel.
                labeling_loss = labeling_loss.mean(0)

                tot_loss = labeling_loss

                # ----- Logging -----
                loss['labeling/L'] = round(labeling_loss.item(), ROUND)

                if config.gradient_accumulation_steps > 1:
                    tot_loss = tot_loss / config.gradient_accumulation_steps

                if config.fp16:
                    raise NotImplementedError()
                else:
                    tot_loss.backward()

                if self.iter_num % config.gradient_accumulation_steps == 0:
                    # ----- Backward for scopes: ['gen'] -----
                    self.step(['gen'])
                    self.zero_grad()

            else:
                raise ValueError()

            if config.display_tqdm:
                display = ', '.join([key + ':' + pretty_string(loss[key]) for key in loss.keys()])
                pbar.set_description_str(display)

            # Print out training information.
            if self.iter_num % config.logging_steps == 0 and config.use_tensorboard:
                self.log_step(loss)

            # Evaluation.
            if self.iter_num % (config.save_steps * config.gradient_accumulation_steps) == 0:
                eval_func = self.get_validation_func()
                no_improvement = eval_func(test=False)

                # Learning rate decay.
                if no_improvement and self.iter_num > config.start_decay:
                    self.no_improvement += 1
                else:
                    self.no_improvement = 0

                if self.no_improvement == config.no_improvement_decay:
                    self.update_lr_by_half()
                    self.no_improvement = 0

    def get_validation_func(self):
        if config.model in ['cprm',
                            'transfertransfo',
                            ]:
            return self.ucpt_evaluate_multiref
        elif config.model in ['persona-labeling',
                              ]:
            self.save_step(['model'])
            return lambda test: True
        else:
            raise ValueError()

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

        peep_num = 50
        for sent_id in range(peep_num):
            print('----- Response -----\n{}'.format(' '.join(batchify['response'][sent_id])))  # For sanity check.
            print('----- Persona probs -----\n{}'.format([round(label.item(), 3) for label in batchify['persona_probs'][sent_id][1:batchify['response_len'][sent_id].item() + 1]]))
            print()
            print('-' * 50)

        self.set_training(mode=True)

        return original_responses, mask_probs

    def ucpt_evaluate_multiref(self, test=False):
        self.set_training(mode=False)

        if test:
            the_set = self.ucpt_test_set
        else:
            the_set = self.ucpt_val_set

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

            if config.model in ['cprm',
                                'transfertransfo',
                                ]:
                with torch.no_grad():
                    preds = self.multi_gpu_forward(input_ids=cls_sep,
                                                   type_ids=type_ids,
                                                   s_len=s_len,
                                                   mode='forward',
                                                   module='model',
                                                   batch_size=batch_size)
            elif config.model == 'backprop':
                # Requires grad
                preds = self.multi_gpu_forward(input_ids=cls_sep,
                                               type_ids=type_ids,
                                               s_len=s_len,
                                               or_input_ids=or_cls_sep,
                                               or_len=or_len,
                                               mode='forward',
                                               module='model',
                                               batch_size=batch_size)
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
