from torch.utils import data
import torch
import os
from collections import defaultdict, Counter
import numpy as np
from functools import reduce
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils.normalize_utils import CONTRACTION_SPACES, CONTRACTION_LEFT_SPACES, CONTRACTION_RIGHT_SPACES
import json
import re
from tqdm import tqdm
from config import Config


config = Config()
UNK_BST = '<unk>'

STOPLIST = stopwords.words('english')
EOSPUNCLIST = list("""!"').?""")
lemmatizer = WordNetLemmatizer()


def convert_text(text):
    """
    Normalize text.
    """
    text = text.lower()
    for x in CONTRACTION_SPACES:
        if x[1] in text:
            text = text.replace(x[1], x[0])
    for x in CONTRACTION_LEFT_SPACES:
        if x[1] in text:
            text = text.replace(x[1], x[0])
    for x in CONTRACTION_RIGHT_SPACES:
        if x[1] in text:
            text = text.replace(x[1], x[0])

    return text


def nltk_convai_normalize(text):
    """
    We use ConvAI2 training data; thus, we need to normalize BST to ConvAI2.

    :param text:
    :return:
    """

    # normalize text.
    text = re.sub(r' *\' *', '\'', text)
    text = convert_text(text)

    # Ending period.
    text = text.strip()
    if text[-1] not in EOSPUNCLIST:
        text = text + ' .'

    return ' '.join(nltk.WordPunctTokenizer().tokenize(text))


class BlendedSkillTalkBase(data.Dataset):
    """The Base dataset."""

    def __init__(self, mode, tokenizer):
        self.mode = mode
        assert self.mode in ['test', 'train', 'valid', None]
        self.tokenizer = tokenizer

    def deduplicate(self, masked_response):
        ret = []
        for token in masked_response:
            if len(ret) == 0 or token != self.tokenizer.unk_token or ret[-1] != self.tokenizer.unk_token:
                ret.append(token)
        return ret

    def lemmatize(self, tokens):
        lem_tokens = []
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = start_idx + 1
            if tokens[start_idx] == self.tokenizer.unk_token:
                lem_tokens.append(self.tokenizer.unk_token)
            else:
                while end_idx < len(tokens):
                    if ' ' in self.tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1]).strip() or tokens[end_idx] == self.tokenizer.unk_token:
                        break
                    else:
                        end_idx += 1

                lem_tokens.extend([lemmatizer.lemmatize(self.tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx]).strip())] * (end_idx - start_idx))
            start_idx = end_idx

        return lem_tokens


class BlendedSkillTalk(BlendedSkillTalkBase):
    """The BlendedSkillTalk dataset."""

    def __init__(self, mode, tokenizer, model):
        super(BlendedSkillTalk, self).__init__(mode, tokenizer)
        self.root = os.path.join('../data', 'blendedskilltalk')

        self.model = model
        if 'no-persona' in self.model:
            self.persona_status = 'without_persona'
        else:
            self.persona_status = 'with_persona'

        if self.model in ['blender-no-persona-cprm',
                          'blender-full-cprm',
                          ]:
            self.cached_masked_file = os.path.join(self.root,
                                                   'cached_{}_{}_mask_probs'.format(mode, self.persona_status))
            self.masked_probs = torch.load(self.cached_masked_file)
        elif self.model in ['blender-full',
                            'blender-no-persona',
                            'transfertransfo',
                            ]:
            pass
        else:
            raise ValueError()

        self.cached_features_file = os.path.join(self.root,
                                                 'cached_{}_{}'.format(mode, self.persona_status))
        if os.path.exists(self.cached_features_file) and not config.overwrite_cache:
            print("Loading features from cached file {}".format(self.cached_features_file))
            self.features = torch.load(self.cached_features_file)
        else:
            self.features = self.read_bst()
            print("Saving features into cached file {}".format(self.cached_features_file))
            torch.save(self.features, self.cached_features_file)

        self.voc_size = len(self.tokenizer)

        print('BlendedSkillTalk data successfully read.')

    def preprocess(self, contexts, intervening_persona, original_response, references):
        contexts = [[self.tokenizer.eos_token if i % 2 == 0 else self.tokenizer.bos_token] + self.tokenizer.tokenize(_context.lower())
                    for i, _context in enumerate(contexts)]
        intervening_persona = [self.tokenizer.tokenize(_intervening_persona.lower()) for _intervening_persona in intervening_persona]

        original_response = self.tokenizer.tokenize(original_response.lower())
        references = [self.tokenizer.tokenize(reference.lower()) for reference in references]
        return contexts, intervening_persona, original_response, references

    def read_bst(self):
        # Process blender predictions
        blender_name = os.path.join(self.root, 'prediction_{}_{}.json'.format(self.mode, self.persona_status))
        with open(blender_name) as fbl:
            original_responses = [nltk_convai_normalize(_blender_data) for _blender_data in json.load(fbl)]

        # Process bst data
        bst_name = os.path.join(self.root, '{}.txt'.format(self.mode))
        bst_data = []
        with open(bst_name) as fbst:
            for line in fbst.readlines():
                line = line.strip()
                assert line.startswith('text:')
                line = line[len('text:'):]

                # New episode.
                if line.startswith('your persona: '):
                    # First persona
                    line = line[len('your persona: '):]
                    persona1 = nltk_convai_normalize(line[:line.index('\\n')])
                    line = line[line.index('\\n') + len('\\n'):]

                    # Second persona
                    assert line.startswith('your persona: ')
                    line = line[len('your persona: '):]
                    persona2 = nltk_convai_normalize(line[:line.index('\\n')])
                    line = line[line.index('\\n') + len('\\n'):]

                    # Save persona
                    personas = (persona1, persona2)

                    # Utterances and response
                    us, r = line.split('\t')[:2]
                    us = [nltk_convai_normalize(u) for u in us.split('\\n')]
                    assert len(us) in [3, 4]
                    if len(us) == 4:
                        us = us[1:]

                    # Update context
                    context = tuple()
                    for u in us:
                        context = (u,) + context  # Reverse.

                    assert r.startswith('labels:')
                    r = nltk_convai_normalize(r[len('labels:'):])

                    # Add to data
                    bst_data.append((context, personas, [r]))

                    # Update context
                    context = (r,) + context  # Reverse.

                else:
                    assert 'your persona: ' not in line

                    # Utterances and response
                    us, r = line.split('\t')[:2]
                    us = [nltk_convai_normalize(u) for u in us.split('\\n')]
                    assert len(us) == 1
                    u = us[0]

                    # Update context
                    context = (u,) + context  # Reverse.

                    assert r.startswith('labels:')
                    r = nltk_convai_normalize(r[len('labels:'):])

                    # Add to data
                    bst_data.append((context, personas, [r]))

                    # Update context
                    context = (r,) + context  # Reverse.

        # Fuse
        features = []
        assert len(bst_data) == len(original_responses)
        for (context, personas, references), original_response in zip(bst_data, original_responses):
            features.append(self.preprocess(contexts=context,
                                            intervening_persona=personas,
                                            original_response=original_response,
                                            references=references))
        return features

    def process_sent(self, contexts, intervening_persona, original_response, references, max_s_len, max_t_len, masked_target):
        # Add persona
        intervening_persona = reduce(lambda x, y: x + y, intervening_persona)
        persona_type_ids = [0] * len(intervening_persona)

        # Add masked response (optional)
        masked_type_ids = []
        if self.model in ['blender-no-persona-cprm',
                          'blender-full-cprm',
                          ]:
            max_total_len = max_s_len + max_t_len
            masked_target = masked_target[:max_t_len]
            masked_type_ids = [3] * len(masked_target)
        elif self.model in ['blender-full',
                            'blender-no-persona',
                            'transfertransfo',
                            ]:
            max_total_len = max_s_len
        else:
            raise ValueError()

        # Add context
        trunc_contexts = [self.tokenizer.bos_token]
        context_type_ids = [2]
        for i, context in enumerate(contexts):
            if len(trunc_contexts) + len(context) + len(intervening_persona) <= max_s_len - 2:
                trunc_contexts = context + trunc_contexts
                if i % 2 == 0:
                    context_type_ids = [1] * len(context) + context_type_ids
                else:
                    context_type_ids = [2] * len(context) + context_type_ids
            else:
                break

        s_len = len(intervening_persona) + len(masked_target) + len(trunc_contexts)

        cls_sep = self.tokenizer.encode([self.tokenizer.cls_token] + intervening_persona + masked_target + trunc_contexts + [self.tokenizer.sep_token],
                                        max_length=max_total_len,
                                        pad_to_max_length=True,
                                        )
        type_ids = [0] + persona_type_ids + masked_type_ids + context_type_ids + [2] * (max_total_len - 1 - s_len)  # [CLS], [SEP], and [PAD]

        # Original response.
        original_response = original_response[:max_t_len - 2]
        or_len = len(original_response)
        or_cls_sep = self.tokenizer.encode([self.tokenizer.cls_token] + original_response + [self.tokenizer.sep_token],
                                           max_length=max_t_len,
                                           pad_to_max_length=True,
                                           )
        outputs = torch.LongTensor(cls_sep), torch.LongTensor([s_len]).squeeze(), torch.LongTensor(type_ids), torch.LongTensor(or_cls_sep), torch.LongTensor([or_len]).squeeze()

        # Add references
        for reference in references:
            reference = reference[:max_t_len]
            ref_len = len(reference)

            ref_cls_sep = self.tokenizer.encode([self.tokenizer.cls_token] + reference + [self.tokenizer.sep_token],
                                                max_length=max_t_len + 2,
                                                pad_to_max_length=True,
                                                )
            outputs = outputs + (torch.LongTensor(ref_cls_sep), torch.LongTensor([ref_len]).squeeze())

        return outputs

    def process_masked(self, response, masked_probs, intervening_persona, contexts):
        assert len(response) >= len(masked_probs)
        response = response[:len(masked_probs)]
        intervening_persona = reduce(lambda x, y: x + y, intervening_persona)
        contexts = reduce(lambda x, y: x + y, contexts)

        lem_response = self.lemmatize(response)
        lem_intervening_persona = self.lemmatize(intervening_persona)
        lem_contexts = self.lemmatize(contexts)
        assert len(lem_response) == len(response)
        assert len(lem_intervening_persona) == len(intervening_persona)
        assert len(lem_contexts) == len(contexts)

        # Mask based on threshold.
        masked_response = []
        for token, lem_token, prob in zip(response, lem_response, masked_probs):
            if prob < config.mask_threshold or lem_token in lem_intervening_persona or lem_token in lem_contexts:
                masked_response.append(token)
            else:
                masked_response.append(self.tokenizer.unk_token)

        # Mask merging.
        masked_response = self.deduplicate(masked_response)

        return masked_response

    def __getitem__(self, index):
        contexts, intervening_persona, original_response, references = self.features[index]
        if self.model in ['blender-no-persona-cprm',
                          'blender-full-cprm',
                          ]:
            masked_probs = self.masked_probs[index]
            masked_target = self.process_masked(response=original_response,
                                                masked_probs=masked_probs,
                                                intervening_persona=intervening_persona,
                                                contexts=contexts)
        elif self.model in ['blender-full',
                            'blender-no-persona',
                            'transfertransfo',
                            ]:
            masked_target = []
        else:
            raise ValueError()

        outputs = self.process_sent(contexts=contexts,
                                    intervening_persona=intervening_persona,
                                    original_response=original_response,
                                    references=references,
                                    max_s_len=config.max_persona_len + config.max_context_len,
                                    max_t_len=config.max_response_len,
                                    masked_target=masked_target)

        return outputs

    def __len__(self):
        return len(self.features)
