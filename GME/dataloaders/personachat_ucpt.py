from torch.utils import data
import torch
import os
import numpy as np
from functools import reduce
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
from config import Config


config = Config()
UNK_PERSONACHAT = '<unk>'

STOPLIST = stopwords.words('english')
EOSPUNCLIST = list("""!"').?""")
lemmatizer = WordNetLemmatizer()


class PersonaChatBase(data.Dataset):
    """The Base dataset."""

    def __init__(self, mode, tokenizer):
        self.mode = mode
        assert self.mode in ['test', 'train', 'valid', None]
        self.tokenizer = tokenizer

    def all_stopwords(self, lemmatized_tokens):
        for lemmatized_token in lemmatized_tokens:
            if lemmatized_token != self.tokenizer.unk_token and (lemmatized_token not in STOPLIST):
                return False
        return True

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


class PersonaChat(PersonaChatBase):
    """The PersonaChat dataset."""

    def __init__(self, mode, tokenizer, model):
        super(PersonaChat, self).__init__(mode, tokenizer)
        self.root = os.path.join('../data', 'personachat-ucpt')
        self.cached_features_file = os.path.join(self.root,
                                                 'cached_{}'.format(mode))
        assert model in [
            'cprm',
            'transfertransfo',
        ]
        self.model = model
        if self.model == 'cprm':
            self.response_to_grads = torch.load(os.path.join(self.root, 'response_to_grads'))

        if os.path.exists(self.cached_features_file) and not config.overwrite_cache:
            print("Loading features from cached file {}".format(self.cached_features_file))
            self.features = torch.load(self.cached_features_file)
        else:
            self.features = self.read_personachat()
            print("Saving features into cached file {}".format(self.cached_features_file))
            torch.save(self.features, self.cached_features_file)

        self.voc_size = len(self.tokenizer)

        print('PersonaChat data successfully read.')

    def read_personachat(self):
        file_name = os.path.join(self.root, 'train.json')
        print(("Reading lines from {}".format(file_name)))
        # Read the file and split into lines
        with_persona_index = 0
        features = []
        with open(file_name) as fr:
            for data in json.load(fr):
                if len(data['persona']) > 0:
                    with_persona_index += 1
                context = tuple()
                for u in data['context']:
                    context = (u, ) + context  # Reverse.
                features.append(self.preprocess(context=context,
                                                persona=data['persona'],
                                                response=data['response']))

        print('{}/{} utterances have their persona index.'.format(with_persona_index, len(features)))
        return features

    def preprocess(self, context, persona, response):
        contexts = [[self.tokenizer.eos_token if i % 2 == 0 else self.tokenizer.bos_token] + self.tokenizer.tokenize(_context)
                    for i, _context in enumerate(context)]
        persona = [] if persona == [] else self.tokenizer.tokenize(persona)

        target = self.tokenizer.tokenize(response)
        return contexts, persona, target

    def process_sent(self, contexts, personas, target, max_s_len, max_t_len, masked_target):
        # Add persona
        personas = reduce(lambda x, y: x + y, personas)
        persona_type_ids = [0] * len(personas)

        # Add masked response (optional)
        masked_type_ids = []
        if self.model == 'cprm':
            max_total_len = max_s_len + max_t_len * 2
            masked_target = masked_target[:max_t_len]
            masked_type_ids = [3] * len(masked_target)
        elif self.model in ['transfertransfo',
                            ]:
            max_total_len = max_s_len + max_t_len
        else:
            raise ValueError()

        # Add context
        if hasattr(config, 'use_context') and not config.use_context:
            contexts = [[]]
        trunc_contexts = [self.tokenizer.bos_token]
        context_type_ids = [2]
        for i, context in enumerate(contexts):
            if len(trunc_contexts) + len(context) + len(personas) <= max_s_len - 2:
                trunc_contexts = context + trunc_contexts
                if i % 2 == 0:
                    context_type_ids = [1] * len(context) + context_type_ids
                else:
                    context_type_ids = [2] * len(context) + context_type_ids
            else:
                break

        # Add target
        target = target[:max_t_len]
        target_type_ids = [2] * len(target)

        s_len = len(personas) + len(masked_target) + len(trunc_contexts)
        t_len = len(target)

        cls_sep = self.tokenizer.encode([self.tokenizer.cls_token] + personas + masked_target + trunc_contexts + target + [self.tokenizer.sep_token],
                                        max_length=max_total_len,
                                        pad_to_max_length=True,
                                        )
        type_ids = [0] + persona_type_ids + masked_type_ids + context_type_ids + target_type_ids + [2] * (max_total_len - 1 - s_len - t_len)  # [CLS], [SEP], and [PAD]

        return torch.LongTensor(cls_sep), torch.LongTensor([s_len]).squeeze(), torch.LongTensor([t_len]).squeeze(), torch.LongTensor(type_ids)

    def process_masked(self, response, persona, grads):

        # Word mask.
        lem_response = self.lemmatize(response)
        lem_persona = self.lemmatize(persona)
        assert len(lem_response) == len(response)
        assert len(lem_persona) == len(persona)

        masked_response = []
        for token, lem_token in zip(response, lem_response):
            if lem_token in lem_persona and lem_token not in STOPLIST + list("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""):
                masked_response.append(self.tokenizer.unk_token)
            else:
                masked_response.append(token)

        for idx, grad in enumerate(grads):
            if grad > config.grad_thres and lem_response[idx] not in STOPLIST + list("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""):
                masked_response[idx] = self.tokenizer.unk_token
                # Whole word mask (right).
                for offset in range(len(lem_response) - idx):
                    if lem_response[idx + offset] == lem_response[idx]:
                        masked_response[idx + offset] = self.tokenizer.unk_token
                    else:
                        break
                # Whole word mask (left).
                for offset in range(idx + 1):
                    if lem_response[idx - offset] == lem_response[idx]:
                        masked_response[idx - offset] = self.tokenizer.unk_token
                    else:
                        break

        # Sentence dropout
        masked_response = self.sentence_dropout(masked_response)

        # Mask merging.
        masked_response = self.deduplicate(masked_response)

        return masked_response

    def sentence_dropout(self, masked_response):

        # Sentence dropout
        segments = []
        start_idx = 0
        for index, token in enumerate(masked_response):
            if self.tokenizer.convert_tokens_to_string([token]).strip() in EOSPUNCLIST:
                end_idx = index + 1
                segments.append((start_idx, end_idx))
                start_idx = end_idx
        if start_idx != len(masked_response):
            segments.append((start_idx, len(masked_response)))

        if config.sentence_dropout:
            no_persona_ids = [idx for (idx, (start_idx, end_idx)) in enumerate(segments) if self.tokenizer.unk_token not in masked_response[start_idx: end_idx]]
            persona_ids = [idx for idx in range(len(segments)) if idx not in no_persona_ids]
            if len(persona_ids) == 0:
                keep_ids = no_persona_ids
            else:
                p = np.exp(-np.arange(len(persona_ids)) / config.tau)  # [0, n - 1]
                p = p / p.sum()
                n_keep = np.random.choice(list(range(len(persona_ids))), p=p)
                keep_ids = list(np.random.choice(persona_ids, size=n_keep)) + no_persona_ids
        else:
            keep_ids = list(range(len(segments)))

        sent_drop = []
        for (idx, segment) in enumerate(segments):
            if idx in keep_ids:
                start_idx, end_idx = segment
                if self.tokenizer.unk_token in masked_response[start_idx: end_idx]:
                    kept_sent = []
                    # Replacement.
                    for tok in masked_response[start_idx: end_idx]:
                        if np.random.uniform() < 0.15:  # Replace prob.
                            kept_sent.append(self.tokenizer.unk_token)
                        else:
                            kept_sent.append(tok)
                else:
                    kept_sent = masked_response[start_idx: end_idx]
                sent_drop.extend(kept_sent)

        return sent_drop

    def __getitem__(self, index):
        contexts, persona, target = self.features[index]
        if self.model == 'cprm':
            response, grads = self.response_to_grads[index]
            masked_target = self.process_masked(response=target,
                                                persona=persona,
                                                grads=grads)
        elif self.model in ['transfertransfo',
                            ]:
            masked_target = []
        else:
            raise ValueError()

        cls_sep, s_len, t_len, type_ids = self.process_sent(contexts=contexts,
                                                            personas=[persona],
                                                            target=target,
                                                            max_s_len=config.max_persona_len + config.max_context_len,
                                                            max_t_len=config.max_response_len,
                                                            masked_target=masked_target)

        return cls_sep, s_len, t_len, type_ids

    def __len__(self):
        return len(self.features)


class PersonaChatPersonaLabeling(PersonaChatBase):
    """The PersonaChat PersonaLabeling dataset."""

    def __init__(self, mode, tokenizer, model):
        super(PersonaChatPersonaLabeling, self).__init__(mode, tokenizer)
        self.model = model
        self.root = os.path.join('../data', 'personachat-ucpt')

        features = PersonaChat('train', tokenizer, 'transfertransfo').features
        self.response_to_grads = torch.load(os.path.join(self.root, 'response_to_grads'))

        if self.mode == 'train':
            if self.model == 'persona-labeling':
                self.features = features
            else:
                raise ValueError()
        elif self.mode in ['valid',
                           'test',
                           ]:
            self.features = features
        else:
            raise ValueError()

    def process_persona_labeling(self, target, persona, max_t_len, grads):
        target = target[:max_t_len]
        grads = grads[:max_t_len]
        t_len = len(target)

        t_cls_sep = self.tokenizer.encode([self.tokenizer.cls_token] + target + [self.tokenizer.sep_token],
                                          max_length=max_t_len + 2,
                                          pad_to_max_length=True,
                                          )
        lem_target = self.lemmatize(target)
        lem_persona = self.lemmatize(persona)
        assert len(lem_target) == len(target)
        assert len(lem_persona) == len(persona)

        p_label = []
        for lem_token in lem_target:
            if lem_token in lem_persona and lem_token not in STOPLIST + list("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""):
                p_label.append(1)
            else:
                p_label.append(0)

        for idx, grad in enumerate(grads):
            if grad > config.grad_thres and lem_target[idx] not in STOPLIST + list("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""):
                p_label[idx] = 1
                # Whole word mask (right).
                for offset in range(len(lem_target) - idx):
                    if lem_target[idx + offset] == lem_target[idx]:
                        p_label[idx + offset] = 1
                    else:
                        break
                # Whole word mask (left).
                for offset in range(idx + 1):
                    if lem_target[idx - offset] == lem_target[idx]:
                        p_label[idx - offset] = 1
                    else:
                        break

        assert len(p_label) == len(target) == t_len
        p_label = [0] + p_label + [0] + [0] * (max_t_len + 2 - 2 - t_len)  # [CLS], [SEP], and [PAD]

        return torch.LongTensor(t_cls_sep), torch.LongTensor([t_len]).squeeze(), torch.LongTensor(p_label)

    def __getitem__(self, index):
        contexts, persona, target = self.features[index]

        response, grads = self.response_to_grads[index]
        t_cls_sep, t_len, p_label = self.process_persona_labeling(target=target,
                                                                  persona=persona,
                                                                  max_t_len=config.max_response_len,
                                                                  grads=grads)

        return t_cls_sep, t_len, p_label

    def __len__(self):
        return len(self.features)


class PersonaChatUCPT(PersonaChatBase):
    """The PersonaChat-UCPT dataset."""

    def __init__(self, mode, tokenizer, model):
        super(PersonaChatUCPT, self).__init__(mode, tokenizer)
        self.root = os.path.join('../data', 'personachat-ucpt')
        self.cached_features_file = os.path.join(self.root,
                                                 'cached_{}'.format(mode))
        self.model = model
        if self.model == 'cprm':
            self.cached_masked_file = os.path.join(self.root,
                                                   'cached_{}_mask_probs_{}'.format(mode, config.grad_thres))
            self.masked_probs = torch.load(self.cached_masked_file)
        elif self.model in ['transfertransfo',
                            'backprop',
                            ]:
            pass
        else:
            raise ValueError()

        if os.path.exists(self.cached_features_file) and not config.overwrite_cache:
            print("Loading features from cached file {}".format(self.cached_features_file))
            self.features = torch.load(self.cached_features_file)
        else:
            self.features = self.read_personachat_ucpt(os.path.join(self.root, '{}.json'.format(self.mode)))
            print("Saving features into cached file {}".format(self.cached_features_file))
            torch.save(self.features, self.cached_features_file)

        self.voc_size = len(self.tokenizer)

        print('PersonaChat-UCPT data successfully read.')

    def preprocess(self, contexts, intervening_persona, original_response, references):
        contexts = [[self.tokenizer.eos_token if i % 2 == 0 else self.tokenizer.bos_token] + self.tokenizer.tokenize(_context)
                    for i, _context in enumerate(contexts)]
        intervening_persona = [self.tokenizer.tokenize(_intervening_persona) for _intervening_persona in intervening_persona]

        original_response = self.tokenizer.tokenize(original_response)
        references = [self.tokenizer.tokenize(reference) for reference in references]
        return contexts, intervening_persona, original_response, references

    def read_personachat_ucpt(self, file_name):
        features = []
        with open(file_name) as f:
            data = json.load(f)
            for _data in data:
                context = tuple()
                for u in _data['context']:
                    context = (u, ) + context  # Reverse.

                features.append(self.preprocess(contexts=context,
                                                intervening_persona=_data['intervening_persona'],
                                                original_response=_data['original_response'],
                                                references=_data['references']))
        return features

    def process_sent(self, contexts, intervening_persona, original_response, references, max_s_len, max_t_len, masked_target):
        # Add persona
        intervening_persona = reduce(lambda x, y: x + y, intervening_persona)
        persona_type_ids = [0] * len(intervening_persona)

        # Add masked response (optional)
        masked_type_ids = []
        if self.model == 'cprm':
            max_total_len = max_s_len + max_t_len
            masked_target = masked_target[:max_t_len]
            masked_type_ids = [3] * len(masked_target)
        elif self.model in ['transfertransfo',
                            'backprop',
                            ]:
            max_total_len = max_s_len
        else:
            raise ValueError()

        # Add context
        if hasattr(config, 'use_context') and not config.use_context:
            contexts = [[]]
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
        if self.model == 'cprm':
            masked_probs = self.masked_probs[index]
            masked_target = self.process_masked(response=original_response,
                                                masked_probs=masked_probs,
                                                intervening_persona=intervening_persona,
                                                contexts=contexts)
        elif self.model in ['transfertransfo',
                            'backprop',
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
