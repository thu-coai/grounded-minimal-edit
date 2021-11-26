# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classes to support Encoder-Decoder architectures """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .modeling_auto import AutoModel, AutoModelWithLMHead
from utils.utils import set_requires_grad

logger = logging.getLogger(__name__)


class PreTrainedSeq2Seq(nn.Module):
    r"""
        :class:`~transformers.PreTrainedSeq2Seq` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and (optionally) another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.
    """

    def __init__(self, model, tokenizer, decoder_max_len, criterionSeq):
        super(PreTrainedSeq2Seq, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.decoder_max_len = decoder_max_len
        self.criterionSeq = criterionSeq

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        tokenizer,
        layers,
        decoder_max_len,
        criterionSeq,
    ):
        r""" Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you need to first set it back in training mode with `model.train()`

        Params:
            encoder_pretrained_model_name_or_path: information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path: information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

                You can specify kwargs sepcific for the encoder and decoder by prefixing the key with `encoder_` and `decoder_` respectively. (e.g. ``decoder_output_attention=True``). The remaining kwargs will be passed to both encoders and decoders.

        Examples::

            model = PreTrainedCVAE.from_pretained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        model = AutoModelWithLMHead.from_pretrained(
            pretrained_model_name_or_path,
            transformer_layers=layers,
            pad_id=tokenizer.pad_token_id,
            is_decoder=True,
        )
        assert model.config.is_decoder

        # Resize token embeddings.
        model.resize_token_embeddings(len(tokenizer))

        model = cls(model, tokenizer, decoder_max_len, criterionSeq)

        return model

    def save_pretrained(self, save_directory):
        """ Save a Seq2Seq model and its configuration file in a format such
        that it can be loaded using `:func:`~transformers.PreTrainedCVAE.from_pretrained`

        We save the encoder' and decoder's parameters in two separate directories.
        """
        self.model.save_pretrained(os.path.join(save_directory, "model"))

    def forward(self, input_ids, type_ids, s_len,
                **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.

        pos = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze_(0).expand(input_ids.shape[0], -1)
        input_mask = pos.le(s_len.unsqueeze(1))
        target_ids = input_ids.masked_fill(input_mask, self.tokenizer.pad_token_id)

        mode = kwargs.pop('mode')
        if mode == 'forward':
            pass
        else:
            raise ValueError()

        if self.training:

            raise ValueError()  # Zero-shot.

        else:
            # For GPT2, it is better to disable batch during inference.
            s_len = s_len.item()

            # Decode
            preds = []
            for i in range(self.decoder_max_len - 1):  # Not shifted.
                if i == 0:
                    # Parallel "encode"
                    logits, presents = self.model(input_ids[:, :s_len + 1],
                                                  token_type_ids=type_ids[:, :s_len + 1],
                                                  past=None)
                    logits = logits[:, -1:]
                else:
                    logits, presents = self.model(input_ids,
                                                  token_type_ids=type_ids[:, s_len + 1: s_len + 2],  # Handled in dataloader
                                                  past=past_key_values)
                    assert logits.shape[1] == 1

                past_key_values = presents

                # Greedy search.
                input_ids = torch.argmax(logits, dim=2)  # shape = (n_batch, 1)
                preds.append(input_ids)

            return torch.cat(preds, dim=1)  # shape = (n_batch, decoder_max_len - 1)


class PreTrainedSeq2SeqBP(nn.Module):
    r"""
        :class:`~transformers.PreTrainedSeq2Seq` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and (optionally) another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.
    """

    def __init__(self, model, tokenizer, decoder_max_len, criterionSeq, num_iter, fgm_iters, stepsize, mix_rate):
        super(PreTrainedSeq2SeqBP, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.decoder_max_len = decoder_max_len
        self.criterionSeq = criterionSeq
        self.num_iter = num_iter
        self.fgm_iters = fgm_iters
        self.stepsize = stepsize
        self.mix_rate = mix_rate

        # Freeze parameters
        set_requires_grad(self.model, requires_grad=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        tokenizer,
        layers,
        decoder_max_len,
        criterionSeq,
        num_iter,
        fgm_iters,
        stepsize,
        mix_rate,
    ):
        r""" Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you need to first set it back in training mode with `model.train()`

        Params:
            encoder_pretrained_model_name_or_path: information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path: information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

                You can specify kwargs sepcific for the encoder and decoder by prefixing the key with `encoder_` and `decoder_` respectively. (e.g. ``decoder_output_attention=True``). The remaining kwargs will be passed to both encoders and decoders.

        Examples::

            model = PreTrainedCVAE.from_pretained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        model = AutoModelWithLMHead.from_pretrained(
            pretrained_model_name_or_path,
            transformer_layers=layers,
            pad_id=tokenizer.pad_token_id,
            is_decoder=True,
        )
        assert model.config.is_decoder

        # Resize token embeddings.
        model.resize_token_embeddings(len(tokenizer))

        model = cls(model, tokenizer, decoder_max_len, criterionSeq, num_iter, fgm_iters, stepsize, mix_rate)

        return model

    def save_pretrained(self, save_directory):
        """ Save a Seq2Seq model and its configuration file in a format such
        that it can be loaded using `:func:`~transformers.PreTrainedCVAE.from_pretrained`

        We save the encoder' and decoder's parameters in two separate directories.
        """
        self.model.save_pretrained(os.path.join(save_directory, "model"))

    def forward(self, input_ids, type_ids, s_len, or_input_ids, or_len,
                **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.

        mode = kwargs.pop('mode')
        if mode == 'forward':
            pass
        else:
            raise ValueError()

        assert not self.training
        # For GPT2, it is better to disable batch during inference.
        s_len = s_len.item()
        or_len = or_len.item()

        # Initialization.
        verbose = [False, True][0]  # TODO
        init_with = ['forward', 'original'][1]  # TODO: in the paper, it should be 'forward'; however, they implement as 'original' in their code.
        sent_logits = self.initial_pass(input_ids=input_ids,
                                        type_ids=type_ids,
                                        s_len=s_len,
                                        or_input_ids=or_input_ids,
                                        init_with=init_with,
                                        verbose=verbose)

        if init_with == 'forward':
            skip_first_back = False
        elif init_with == 'original':
            skip_first_back = True
        else:
            raise ValueError()

        candidate_list = []
        for t in range(self.num_iter):
            if t == 0 and skip_first_back:
                backward_logits = sent_logits
            else:
                backward_logits = self.backward_pass(
                    sent_logits,
                    or_input_ids=or_input_ids,
                    verbose=verbose,
                )

            sent_logits, forward_text = self.forward_pass(
                backward_logits=backward_logits,
                input_ids=input_ids,
                type_ids=type_ids,
                s_len=s_len,
                or_len=or_len,
                verbose=verbose,
            )
            candidate_list.append(forward_text)
        if verbose:
            print('=' * 100)

        return candidate_list[-1]  # No ranking.

    def backward_pass(self, sent_logits, or_input_ids, verbose):
        device = sent_logits.device

        # Set logits to a list just for ease of programming and experimentation
        sent_logits = [sent_logits]

        if verbose:
            print("[Before backward]: ", self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(torch.argmax(sent_logits[0][0], dim=1))))

        # Accumuated gradients w.r.t the logits
        grad_accumulator = [(np.zeros(p.shape).astype("float32")) for p in sent_logits]

        # Accumulate perturbations for num_backward_iters
        for i in range(self.fgm_iters):
            if verbose:
                print("\n-------Iteration------- ", i + 1)

            # Compute the perturbed sent_logits
            # TODO: what is volatile?
            curr_perturbation = [Variable(torch.from_numpy(p_).to(device=device), requires_grad=True, volatile=False) for p_ in grad_accumulator]
            perturbed_logits = list(map(lambda a, b: a + b, sent_logits, curr_perturbation))

            # Compute the norms of the sent_logits for normalizing the gradients later
            perturbed_logits_norms_all = [torch.norm(p_) for index, p_ in enumerate(perturbed_logits)]

            # Compute loss
            loss = self.criterionSeq(perturbed_logits[0], or_input_ids[:, 1:]).sum(0)
            if verbose:
                print("loss: %.4f" % (loss.data.cpu().numpy()))

            # Compute gradients
            loss.backward()

            # Compute gradient norms
            grad_norms_all = [(torch.norm(p_.grad) + 1e-15) for index, p_ in enumerate(curr_perturbation)]
            # Normalize and scale the gradients
            grad = [
                - self.stepsize * (p_.grad / grad_norms_all[index] * perturbed_logits_norms_all[index]).data.cpu().numpy()
                for index, p_ in enumerate(curr_perturbation)
            ]

            # Accumulate gradients
            grad_accumulator = list(map(lambda a, b: a + b, grad, grad_accumulator))

            # Reset gradients
            for p_ in curr_perturbation:
                p_.grad.data.zero_()

            # Remove logits from the graph.  TODO: so soft logits is only used once for counterfactual in their implementation.
            new_logits = []
            for p_ in sent_logits:
                new_logits.append(p_.detach())
            sent_logits = new_logits

            if verbose:  # inspect the temporary text after the backward pass
                _grad_accumulator = [Variable(torch.from_numpy(p_).to(device=device), requires_grad=True, volatile=False) for p_ in grad_accumulator]
                _pert_logits = list(map(lambda a, b: a + b, sent_logits, _grad_accumulator))
                print("[Backward]: ", self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(torch.argmax(_pert_logits[0][0], dim=1))))

        # Apply the accumulated gradients to the logits
        grad_accumulator = [Variable(torch.from_numpy(p_).to(device=device), requires_grad=True, volatile=False) for p_ in grad_accumulator]
        pert_logits = list(map(lambda a, b: a + b, sent_logits, grad_accumulator))

        return pert_logits[0]

    def forward_pass(self, backward_logits, input_ids, type_ids, s_len, or_len, verbose):

        last_embeds = None
        logits_so_far = []
        logits_so_far_complete = []
        for i in range(self.decoder_max_len - 1):

            if i == 0:
                # Parallel "encode"
                forward_logits, presents = self.model(input_ids[:, :s_len + 1],
                                                      token_type_ids=type_ids[:, :s_len + 1],
                                                      past=None)
                forward_logits = forward_logits[:, -1:]
            else:
                forward_logits, presents = self.model(inputs_embeds=last_embeds,
                                                      token_type_ids=type_ids[:, s_len + 1: s_len + 2],
                                                      past=past_key_values)
                forward_logits = forward_logits[:, -1:]  # the original implementation has a temperature here, which is not stated in the paper

            past_key_values = presents
            if i < or_len + 1:  # Including sep_token
                # Mix backward logits and forward logits.
                sent_logits = self.mix_rate * forward_logits + (1 - self.mix_rate) * backward_logits[:, i:i+1, :]
            else:
                # Continue to complete the text
                sent_logits = forward_logits

            if i < or_len + 1:  # Including sep_token
                logits_so_far.append(sent_logits)
            logits_so_far_complete.append(sent_logits)

            # Use a small temperature (0.1) so that the soft token representation is closer to a one-hot representation
            last_embeds = torch.matmul(torch.softmax(sent_logits / 0.1, dim=2), self.model.transformer.wte.weight)  # Temperate is also hard-coded in the original implementation.

        logits_so_far = torch.cat(logits_so_far, dim=1)
        logits_so_far_complete = torch.cat(logits_so_far_complete, dim=1)

        # Sample a text, and only extract the first sentence
        forward_text = torch.argmax(logits_so_far_complete, dim=2)
        if verbose:
            print("[Forward]: ", self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(forward_text[0])))

        # We return logits_so_far_complete instead because the KL objective in
        # counterfactual reasoning will not update parts longer than os_len
        return logits_so_far_complete, forward_text

    def initial_pass(self, input_ids, type_ids, s_len, or_input_ids, init_with, verbose):
        # Decode
        sent_logits = []
        all_input_ids = []
        for i in range(self.decoder_max_len - 1):  # Not shifted.
            if i == 0:
                # Parallel "encode"
                logits, presents = self.model(input_ids[:, :s_len + 1],
                                              token_type_ids=type_ids[:, :s_len + 1],
                                              past=None)
                logits = logits[:, -1:]
            else:
                logits, presents = self.model(input_ids,
                                              token_type_ids=type_ids[:, s_len + 1: s_len + 2],  # Handled in dataloader
                                              past=past_key_values)

            past_key_values = presents

            if init_with == 'forward':
                # Greedy search.
                input_ids = torch.argmax(logits, dim=2)  # shape = (n_batch, 1)
            elif init_with == 'original':
                input_ids = or_input_ids[:, i+1:i+2]  # because 1) for the next position and 2) or_input_ids has cls_token
            else:
                raise ValueError()

            sent_logits.append(logits)
            all_input_ids.append(input_ids)

        sent_logits = torch.cat(sent_logits, dim=1)  # shape = (n_batch, decoder_max_len - 1, V)
        all_input_ids = torch.cat(all_input_ids, dim=1)

        if verbose:
            print("[Initial]: ", self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(torch.argmax(sent_logits[0], dim=1))))
            print("[Input ids]: ", self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(all_input_ids[0])))
            print('-' * 50)
        return sent_logits


class PreTrainedSequenceLabeling(nn.Module):
    r"""
        :class:`~transformers.PreTrainedSequenceLabeling` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as the language model when created with the
        `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.
    """

    def __init__(self, encoder, n_label, label_weight, criterionSeq):
        super(PreTrainedSequenceLabeling, self).__init__()
        self.encoder = encoder
        self.n_label = n_label
        if label_weight is not None:
            self.register_buffer('label_weight', label_weight)
        self.criterionSeq = criterionSeq

        self.cls = nn.Linear(self.encoder.config.hidden_size, self.n_label)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        layers,
        tokenizer,
        n_label,
        label_weight,
        criterionSeq,
    ):
        r""" Instantiates a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you need to first set it back in training mode with `model.train()`

        Params:
            encoder_pretrained_model_name_or_path: information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path: information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

                You can specify kwargs sepcific for the encoder and decoder by prefixing the key with `encoder_` and `decoder_` respectively. (e.g. ``decoder_output_attention=True``). The remaining kwargs will be passed to both encoders and decoders.

        Examples::

            model = PreTrainedEncoderDecoder.from_pretained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            transformer_layers=layers,
            pad_id=tokenizer.pad_token_id,
            is_decoder=False,  # TODO: modified by Chen Wu.
        )
        assert not model.config.is_decoder

        model.resize_token_embeddings(len(tokenizer))

        model = cls(model, n_label, label_weight, criterionSeq)

        return model

    def save_pretrained(self, save_directory):
        """ Save a Seq2Seq model and its configuration file in a format such
        that it can be loaded using `:func:`~transformers.PreTrainedEncoderDecoder.from_pretrained`

        We save the encoder' and decoder's parameters in two separate directories.
        """
        self.model.save_pretrained(os.path.join(save_directory, "model"))

    def forward(self, encoder_input_ids, encoder_labels=None, **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
                OR: ``torch.LongTensor`` of shape ``(batch_size, sequence_length, vocabulary_size)``
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.

        mode = kwargs.pop('mode')
        if mode == 'forward':
            pass
        elif mode == 'persona_prob':
            pass
        else:
            raise ValueError()

        B, T = encoder_input_ids.shape

        encoder_outputs = self.encoder(input_ids=encoder_input_ids)
        encoder_hidden_states = encoder_outputs[0]  # shape = (n_batch, max_len, hidden_size)

        logits = self.cls(encoder_hidden_states)  # shape = (n_batch, max_len, n_label)

        if self.training:

            # Compute loss
            labeling_loss = F.cross_entropy(logits.contiguous().view(B * T, self.n_label),
                                            encoder_labels.view(B * T),
                                            weight=self.label_weight,
                                            reduction='none').contiguous().view(B, T)  # shape = (n_batch, max_len)
            label_mask = encoder_input_ids.eq(self.criterionSeq.pad) | encoder_input_ids.eq(self.criterionSeq.bos) | encoder_input_ids.eq(self.criterionSeq.eos)  # shape = (n_batch, max_len)
            labeling_loss = (labeling_loss * (1 - label_mask.float())).sum(1)  # shape = (n_batch, )
            return labeling_loss

        else:
            label_mask = encoder_input_ids.eq(self.criterionSeq.pad) | encoder_input_ids.eq(self.criterionSeq.bos) | encoder_input_ids.eq(self.criterionSeq.eos)  # shape = (n_batch, max_len)

            if mode == 'forward':
                preds = torch.argmax(logits, dim=2)  # shape = (n_batch, max_len): 0 is not persona, 1 is persona.
                preds = preds.masked_fill(label_mask, 0)  # Do not label bos, eos, and pad as persona.
                return preds
            elif mode == 'persona_prob':
                persona_probs = torch.softmax(logits, dim=2)[:, :, 1]  # shape = (n_batch, max_len): 0 is not persona, 1 is persona.
                persona_probs = persona_probs.masked_fill(label_mask, 0)  # Do not label bos, eos, and pad as persona.
                return persona_probs
            else:
                raise ValueError()




