import os
import torch
import glob


class Config(object):

    def __init__(self):

        self.task_id = [0,  # PersonaChat.
                        ][0]

        self.task = ['persona-chat',  # 0
                     ][self.task_id]
        self.model = ['cprm',  # 0
                      'transfertransfo',  # 1
                      'backprop',  # 2
                      'persona-labeling',  # 3
                      ][0]

        self.model_name_or_path = ['gpt2'][0]

        # Training configuration.
        self.use_local = True
        self.max_grad_norm = 1.0
        self.gen_lr = 5e-5
        self.max_response_len = 32
        self.max_persona_len = 16
        self.max_context_len = 64
        self.beam_size = 1

        self.overwrite_cache = False
        self.fp16 = False
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.start_decay = 0
        self.optimizer = 'adam'
        if self.model in ['persona-labeling',
                          ]:
            self.gpu_ids = [0]
            self.train_batch_size = 32
            self.eval_batch_size = 32
        else:
            self.gpu_ids = [0]
            self.train_batch_size = 32
            self.eval_batch_size = len(self.gpu_ids)
        self.gradient_accumulation_steps = 2
        self.bucket = [False, True][0]
        self.layers = 'default'
        self.display_tqdm = [False, True][0]

        # Miscellaneous.
        self.num_workers = 8
        self.use_tensorboard = True
        self.ROUND = 4
        if self.model in ['persona-labeling',
                          ]:
            self.seed = 0
        else:
            self.seed = [0, 1, 2, 3, 4, 5][3]
        self.gpu = torch.cuda.is_available()

        # Steps.
        self.logging_steps = 50
        self.max_decay_num = 2
        self.no_improvement_decay = 3
        self.save_steps = 500

        # Method-related.
        self.grad_thres, self.label_occurance = (3, (1418321, 40691))

        if self.model == 'cprm':
            self.mask_threshold = 0.5
            self.smooth_eps = 0.1
            self.tau = 3
            # Ablations.
            self.sentence_dropout, self.use_context = True, True
            self.ablation = ['none', 'sentence_dropout', 'context'][0]
            if self.ablation == 'none':
                pass
            elif self.ablation == 'sentence_dropout':
                self.sentence_dropout = False
            elif self.ablation == 'context':
                self.use_context = False
            else:
                raise ValueError()
        elif self.model == 'transfertransfo':
            pass
        elif self.model == 'backprop':
            self.gpu_ids = [0]
            self.eval_batch_size = len(self.gpu_ids)
            self.num_iter = 10
            self.fgm_iters = 10
            self.stepsize = 4e-4
            self.mix_rate = [0.75, 0.8, 0.85][0]
        elif self.model in ['persona-labeling',
                            ]:
            self.model_name_or_path = 'gpt2'
            self.save_steps = 2000
        else:
            raise ValueError()

        # Directories.
        if self.use_local:
            self.DIR = '../pretrained_transformer_weights'
            self.model_name_or_path = os.path.join(self.DIR, self.model_name_or_path)
        self.log_dir = self.model_specific_dir('outputs/logs')
        remove_all_under(self.log_dir)
        self.save_model_dir = self.model_specific_dir('outputs/saved_model')
        self.sample_dir = self.model_specific_dir('outputs/sampled_results')
        self.tmp_dir = self.model_specific_dir('outputs/temp_results')

    def model_specific_dir(self, root):
        """ model-normalization """

        name_components = [
            self.task,
            self.model,
        ]

        if self.model == 'cprm':
            name_components.append('smooth_eps{}'.format(self.smooth_eps))
            name_components.append('grad_thres{}'.format(self.grad_thres))
            if self.tau != 1:
                name_components.append('tau{}'.format(self.tau))
            if not self.sentence_dropout:
                name_components.append('no_sentence_dropout')
            if not self.use_context:
                name_components.append('no_context')
            name_components.append('{}'.format(self.seed))
        elif self.model == 'transfertransfo':
            name_components.append('{}'.format(self.seed))
        elif self.model == 'backprop':
            name_components.append('mix_rate{}'.format(self.mix_rate))
            name_components.append('{}'.format(self.seed))
        elif self.model in ['persona-labeling',
                            ]:
            assert self.seed == 0
            name_components.append('grad_thres{}'.format(self.grad_thres))
        else:
            raise ValueError()

        ret = os.path.join(root, '-'.join(name_components))
        if not os.path.exists(ret):
            os.mkdir(ret)
        return ret


def remove_all_under(directory):
    for file in glob.glob(os.path.join(directory, '*')):
        os.remove(file)
