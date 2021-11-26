import os
import shutil
from dataloaders.blendedskilltalk import BlendedSkillTalk, BlendedSkillTalkBase
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Experiment import Experiment
from config import Config

config = Config()


class CacheBSTLabeling(BlendedSkillTalkBase):
    """The CacheBSTLabeling dataset."""

    def __init__(self, tokenizer, dataset):
        super(CacheBSTLabeling, self).__init__(None, tokenizer)

        self.dataset = dataset

        # With / without persona
        if self.dataset.split('-')[1] == 'with_persona':
            base_data = 'blender-full'
        elif self.dataset.split('-')[1] == 'without_persona':
            base_data = 'blender-no-persona'
        else:
            raise ValueError()

        # Split
        split = self.dataset.split('-')[2]

        self.features = BlendedSkillTalk(split, tokenizer, base_data).features

    def process_response(self, response, max_t_len):
        response = response[:max_t_len]
        t_len = len(response)

        t_cls_sep = self.tokenizer.encode([self.tokenizer.cls_token] + response + [self.tokenizer.sep_token],
                                          max_length=max_t_len + 2,
                                          pad_to_max_length=True,
                                          )

        return torch.LongTensor(t_cls_sep), torch.LongTensor([t_len]).squeeze()

    def __getitem__(self, index):
        contexts, intervening_persona, original_response, references = self.features[index]
        t_cls_sep, t_len = self.process_response(response=original_response,
                                                 max_t_len=config.max_response_len)

        return t_cls_sep, t_len

    def __len__(self):
        return len(self.features)


def persona_labeling():
    # Building the wrapper
    wrapper = Experiment()

    if config.model == 'persona-labeling':
        # Three corpora.
        datasets = [#'blendedskilltalk-valid',
                    'blendedskilltalk-with_persona-test',
                    'blendedskilltalk-without_persona-test',
                    ]
        SAVE_FILES = {
            'blendedskilltalk-valid': '../data/blendedskilltalk/cached_valid_mask_probs',
            'blendedskilltalk-with_persona-test': '../data/blendedskilltalk/cached_test_with_persona_mask_probs',
            'blendedskilltalk-without_persona-test': '../data/blendedskilltalk/cached_test_without_persona_mask_probs',
        }
    else:
        raise ValueError()

    for dataset in datasets:
        responses = []
        response_lens = []
        data = CacheBSTLabeling(wrapper.tokenizer, dataset=dataset)
        for index in tqdm(range(len(data))):
            t_cls_sep, t_len = data[index]
            responses.append(t_cls_sep)
            response_lens.append(t_len)
        responses = torch.stack(responses, dim=0)
        response_lens = torch.stack(response_lens, dim=0)

        original_responses, mask_probs = wrapper.persona_labeling(responses=responses, response_lens=response_lens)
        assert len(original_responses) == len(mask_probs) == responses.shape[0] == response_lens.shape[0]
        torch.save(mask_probs, SAVE_FILES[dataset])


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    mode = ['persona-labeling'][0]
    if mode == 'persona-labeling':
        persona_labeling()
    else:
        raise ValueError()
