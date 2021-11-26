import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SeqLoss(nn.Module):
    def __init__(self, voc_size, pad, bos, eos, unk, prob):
        super(SeqLoss, self).__init__()
        self.voc_size = voc_size
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.prob = prob
        word_weight = torch.ones(voc_size)
        word_weight[pad] = 0.
        self.register_buffer("word_weight", word_weight)

    def forward(self, inputs, gts, keep_batch=False):
        """
        :param inputs: (?, T, V)
        :param gts: (?, T)
        :param keep_batch: bool.
        :return: Scalar or (?).
        """
        if inputs.shape[0] == 0:
            raise ValueError()

        assert inputs.shape[:-1] == gts.shape
        if not keep_batch:
            if self.prob:
                xent = F.nll_loss(input=torch.log(torch.clamp(inputs.contiguous().view(-1, self.voc_size), min=1e-10)),
                                  target=gts.view(-1),
                                  weight=self.word_weight)
            else:
                xent = F.cross_entropy(input=inputs.contiguous().view(-1, self.voc_size),
                                       target=gts.view(-1),
                                       weight=self.word_weight)
            return xent
        else:
            T = inputs.shape[-2]
            stuct_shape = list(inputs.shape[:-2])
            if self.prob:
                xent = F.nll_loss(input=torch.log(torch.clamp(inputs.contiguous().view(-1, self.voc_size), min=1e-10)),
                                  target=gts.view(-1),
                                  weight=self.word_weight,
                                  reduction='none')
            else:
                xent = F.cross_entropy(input=inputs.contiguous().view(-1, self.voc_size),
                                       target=gts.view(-1),
                                       weight=self.word_weight,
                                       reduction='none')
            xent = xent.view(stuct_shape + [T])  # shape = (?, T)
            xent = xent.sum(-1)  # shape = (?)
            return xent
