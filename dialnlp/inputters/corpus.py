#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved.
#
# File: dialnlp/inputters/corpus.py
# Date: 2018/12/06 15:34:50
# Author: chenchaotao@baidu.com
#
################################################################################

import os
import torch

from tqdm import tqdm
from dialnlp.inputters.field import tokenize
from dialnlp.inputters.field import TextField
from dialnlp.inputters.field import NumberField
from dialnlp.inputters.dataset import Dataset


class Corpus(object):
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        prepared_data_file = f"{data_prefix}_{max_vocab_size}.data.pt"
        prepared_vocab_file = f"{data_prefix}_{max_vocab_size}.vocab.pt"
        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.fields = {}
        self.filter_pred = None
        self.sort_fn = None
        self.data = None

    def load_data(self, prepared_data_file):
        print(f"Loading prepared data from {prepared_data_file} ...")
        data = torch.load(prepared_data_file)
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}
        print("Number of examples:",
              " ".join(f"{k.upper()}-{len(v)}" for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        print(f"Loading prepared vocab from {prepared_vocab_file} ...")
        vocab_dict = torch.load(prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join(f"{name.upper()}-{field.vocab_size}" for name, field in
                       self.fields.items() if isinstance(field, TextField)))

    def read_data(self, data_file, data_type):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print(f"Building vocabulary of field {name.upper()} ...")
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        examples = []
        ignored = 0
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                example[name] = self.fields[name].numericalize(strings)
            if self.filter_pred is None or self.filter_pred(example):
                examples.append(example)
            else:
                ignored += 1
        if self.sort_fn is not None:
            print(f"Sorting examples ...")
            examples = self.sort_fn(examples)
        print(f"Built {len(examples)} examples ({ignored} ignored)")
        return examples

    def build(self):
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, self.data_prefix+".train")
        valid_file = os.path.join(self.data_dir, self.data_prefix+".valid")
        test_file = os.path.join(self.data_dir, self.data_prefix+".test")

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")
        vocab = self.build_vocab(train_raw)

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saving prepared vocab ...")
        torch.save(vocab, self.prepared_vocab_file)
        print(f"Saved prepared vocab to '{self.prepared_vocab_file}'")

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print(f"Saved prepared data to '{self.prepared_data_file}'")

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError(f"Unsported data type: {data_type}!")

    def transform(self, data_file, batch_size, data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader


class SrcTgtCorpus(Corpus):
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False):
        super(SrcTgtCorpus, self).__init__(data_dir=data_dir,
                                           data_prefix=data_prefix,
                                           min_freq=min_freq,
                                           max_vocab_size=max_vocab_size)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab

        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)

        self.fields = {'src': self.SRC, 'tgt': self.TGT}

        def src_filter_pred(src):
            return min_len <= len(src)-2 <= max_len

        def tgt_filter_pred(tgt):
            return min_len <= len(tgt)-2 <= max_len

        self.filter_pred = lambda ex: src_filter_pred(
            ex['src']) and tgt_filter_pred(ex['tgt'])

    def load(self):
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()
        self.load_vocab(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)

        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]

    def read_data(self, data_file, data_type="train"):
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt = line.strip().split('\t')[:2]
                data.append({'src': src, 'tgt': tgt})
        return data
