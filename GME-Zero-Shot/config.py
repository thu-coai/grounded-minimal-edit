import os
from functools import reduce
import torch
import glob
import math
import numpy as np


class Config(object):

    def __init__(self):

        self.task_id = [0,  # PersonaChat.
                        ][0]

        self.task = ['blendedskilltalk',  # 0
                     ][self.task_id]
        self.model = ['blender-no-persona-cprm',  # 0
                      'blender-full-cprm',  # 1
                      'blender-full',  # 2
                      'blender-no-persona',  # 3
                      'transfertransfo',  # 4
                      'persona-labeling',  # 5
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
        self.gpu_ids = [0]
        self.train_batch_size = 32
        self.eval_batch_size = len(self.gpu_ids)
        self.gradient_accumulation_steps = 1
        self.bucket = [False, True][0]
        self.layers = 'default'
        self.display_tqdm = [False, True][1]

        # Miscellaneous.
        self.num_workers = 0
        self.use_tensorboard = True
        self.ROUND = 4
        self.seed = 0
        self.gpu = torch.cuda.is_available()

        # Steps.
        self.logging_steps = 50
        self.max_decay_num = 2
        self.no_improvement_decay = 3
        self.save_steps = 500

        # Method-related.
        self.grad_thres, self.label_occurance = (3, (1418321, 40691))

        if self.model in ['blender-no-persona-cprm',
                          'blender-full-cprm',
                          ]:
            self.mask_threshold = 0.75
            self.tau = 3
        elif self.model == 'blender-full':
            pass
        elif self.model == 'blender-no-persona':
            pass
        elif self.model == 'transfertransfo':
            pass
        elif self.model == 'persona-labeling':
            pass
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

        if self.model in ['blender-no-persona-cprm',
                          'blender-full-cprm',
                          ]:
            name_components[0] = 'persona-chat'  # zero-shot
            name_components[1] = 'cprm'  # zero-shot
            name_components.append('smooth_eps0.1')
            name_components.append('grad_thres{}'.format(self.grad_thres))
            if self.tau != 1:
                name_components.append('tau{}'.format(self.tau))
            name_components.append('{}'.format(self.seed))
        elif self.model == 'blender-full':
            pass
        elif self.model == 'blender-no-persona':
            pass
        elif self.model == 'transfertransfo':
            name_components[0] = 'persona-chat'  # zero-shot
            name_components.append('0')
        elif self.model == 'persona-labeling':
            name_components[0] = 'persona-chat'  # zero-shot
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
