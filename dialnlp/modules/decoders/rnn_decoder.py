#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved.
#
# File: dialnlp/decoders/rnn_decoder.py
# Date: 2018/11/10 15:53:36
# Author: chenchaotao@baidu.com
#
################################################################################

import torch
import torch.nn as nn

from dialnlp.modules.attention import Attention
from dialnlp.modules.decoders.state import RNNDecoderState
from dialnlp.utils.misc import sequence_mask


class RNNDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2
        self.memory_size = memory_size or hidden_size
        self.feature_size = feature_size
        self.dropout = dropout

        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_size

        if self.feature_size is not None:
            self.rnn_input_size += self.feature_size

        if self.attn_mode is not None:
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.rnn_input_size += self.memory_size
            self.out_input_size += self.memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )

    def initialize_state(self,
                         hidden,
                         feature=None,
                         attn_memory=None,
                         attn_mask=None,
                         memory_lengths=None):
        if self.feature_size is not None:
            assert feature is not None

        if self.attn_mode is not None:
            assert attn_memory is not None

        if memory_lengths is not None and attn_mask is None:
            max_len = attn_memory.size(1)
            attn_mask = sequence_mask(memory_lengths, max_len).eq(0)

        init_state = RNNDecoderState(
            hidden=hidden,
            feature=feature,
            attn_memory=attn_memory,
            attn_mask=attn_mask,
        )
        return init_state

    def decode(self, input, state, num_valid=None):
        """
        num_valid: number of valid input, just used in training phrase
        """
        hidden = state.get("hidden", num_valid)
        if num_valid is not None:
            hidden = hidden.clone()

        rnn_input_list = []
        out_input_list = []

        if num_valid is not None:
            input = input[:num_valid]

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        rnn_input_list.append(input)

        if self.feature_size is not None:
            feature = state.get("feature", num_valid).unsqueeze(1)
            rnn_input_list.append(feature)

        attn = None
        if self.attn_mode is not None:
            attn_memory = state.get("attn_memory", num_valid)
            attn_mask = state.get("attn_mask", num_valid)
            query = hidden[-1].unsqueeze(1)
            weighted_context, attn = self.attention(query=query,
                                                    memory=attn_memory,
                                                    mask=attn_mask)
            rnn_input_list.append(weighted_context)
            out_input_list.append(weighted_context)

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)
        out_input_list.append(rnn_output)

        out_input = torch.cat(out_input_list, dim=-1)
        if num_valid is None:
            state.hidden = new_hidden
            output = self.output_layer(out_input)
        else:
            state.hidden[:, :num_valid] = new_hidden
            output = out_input

        return output, state, attn

    def forward(self, inputs, state):
        inputs, lengths = inputs
        batch_size, max_dec_len = inputs.size()

        outputs = inputs.new_zeros(
            size=(batch_size, max_dec_len, self.out_input_size),
            dtype=torch.float)

        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:, i]
            output, state, _ = self.decode(dec_input, state, num_valid)
            outputs[:num_valid, i] = output.squeeze(1)

        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        outputs = outputs.index_select(0, inv_indices)
        outputs = self.output_layer(outputs)
        return outputs, state
