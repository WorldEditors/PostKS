#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved.
#
# File: run_dssm.py
# Date: 2018/12/14 11:49:51
# Author: chenchaotao@baidu.com
#
################################################################################

import os
import json
import logging
import argparse
import torch
from datetime import datetime

from dialnlp.inputters.corpus import SrcTgtCorpus
from dialnlp.models.dssm import DSSM
from dialnlp.utils.engine import Trainer
from dialnlp.utils.engine import evaluate
from dialnlp.utils.misc import str2bool


parser = argparse.ArgumentParser()

# Data
data_arg = parser.add_argument_group("Data")
data_arg.add_argument("--data_dir", type=str, default="./data/toy/")
data_arg.add_argument("--save_dir", type=str, default="./outputs/toy/dssm/")
data_arg.add_argument("--embed_file", type=str, default=None)
# data_arg.add_argument("--embed_file", type=str,
#                       default="./embeddings/sgns.weibo.300d.txt")

# Network
net_arg = parser.add_argument_group("Network")
net_arg.add_argument("--embed_size", type=int, default=300)
net_arg.add_argument("--hidden_size", type=int, default=800)
net_arg.add_argument("--bidirectional", type=str2bool, default=True)
net_arg.add_argument("--max_vocab_size", type=int, default=40000)
net_arg.add_argument("--min_len", type=int, default=5)
net_arg.add_argument("--max_len", type=int, default=50)
net_arg.add_argument("--num_layers", type=int, default=1)
net_arg.add_argument("--share_vocab", type=str2bool, default=True)
net_arg.add_argument("--tie_embedding", type=str2bool, default=True)
net_arg.add_argument("--with_project", type=str2bool, default=False)

# Training / Testing
train_arg = parser.add_argument_group("Training")
train_arg.add_argument("--optimizer", type=str, default="Adam")
train_arg.add_argument("--learning_rate", type=float, default=0.0002)
train_arg.add_argument("--grad_clip", type=float, default=5.0)
train_arg.add_argument("--dropout", type=float, default=0.3)
train_arg.add_argument("--num_epochs", type=int, default=10)
train_arg.add_argument("--lr_decay", type=str2bool, default=False)
train_arg.add_argument("--use_embed", type=str2bool, default=True)

# MISC
misc_arg = parser.add_argument_group("Misc")
misc_arg.add_argument("--gpu", type=int, default=-1)
misc_arg.add_argument("--log_steps", type=int, default=50)
misc_arg.add_argument("--valid_steps", type=int, default=100)
misc_arg.add_argument("--batch_size", type=int, default=32)
misc_arg.add_argument("--ckpt", type=str)
misc_arg.add_argument("--test", action="store_true")
misc_arg.add_argument("--check", action="store_true")
misc_arg.add_argument("--infer", action="store_true")
misc_arg.add_argument("--infer_file", type=str, default=None)

config = parser.parse_args()


def main():
    if config.check:
        config.save_dir = "./tmp/"

    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    device = config.gpu
    torch.cuda.set_device(device)

    # Data definition
    corpus = SrcTgtCorpus(data_dir=config.data_dir,
                          data_prefix="dial",
                          min_freq=0,
                          max_vocab_size=config.max_vocab_size,
                          min_len=config.min_len,
                          max_len=config.max_len,
                          embed_file=config.embed_file,
                          share_vocab=config.share_vocab)
    corpus.load()

    train_iter = corpus.create_batches(
        config.batch_size, "train", shuffle=True, device=device)
    valid_iter = corpus.create_batches(
        config.batch_size, "valid", shuffle=False, device=device)
    test_iter = corpus.create_batches(
        config.batch_size, "test", shuffle=False, device=device)

    # Model Definition
    model = DSSM(src_vocab_size=corpus.SRC.vocab_size,
                 tgt_vocab_size=corpus.TGT.vocab_size,
                 embed_size=config.embed_size,
                 hidden_size=config.hidden_size,
                 padding_idx=corpus.padding_idx,
                 num_layers=config.num_layers,
                 bidirectional=config.bidirectional,
                 tie_embedding=config.tie_embedding,
                 margin=None,
                 with_project=config.with_project,
                 dropout=config.dropout,
                 use_gpu=config.use_gpu)

    model_name = model.__class__.__name__

    if config.test and config.ckpt:
        print(model)
        model.load(config.ckpt)

        print("Testing ...")
        print(evaluate(model, test_iter).report_cum())

    elif config.infer and config.ckpt and config.infer_file:
        from dialnlp.inputters.dataset import Dataset
        print(model)
        model.load(config.ckpt)
        infer_data = corpus.read_data(config.infer_file, data_type="test")
        infer_examples = corpus.build_examples(infer_data)
        infer_data = Dataset(infer_examples)
        infer_iter = infer_data.create_batches(batch_size=config.batch_size,
                                               shuffle=False,
                                               device=device)

        results = []
        from tqdm import tqdm
        for inputs in tqdm(infer_iter):
            score = model.score(inputs).tolist()
            src = corpus.SRC.denumericalize(inputs.src[0])
            tgt = corpus.TGT.denumericalize(inputs.tgt[0])
            results += list(zip(src, tgt, score))

        infer_result_file = "./infer.result"
        with open(infer_result_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write("\t".join(map(str, result)) + "\n")
        print(f"Saved infer results to '{infer_result_file}'")

    else:
        # Load Word Embeddings
        if config.use_embed and config.embed_file is not None:
            model.src_encoder.embedder.load_embeddings(
                corpus.SRC.embeddings, scale=0.03)
            model.tgt_encoder.embedder.load_embeddings(
                corpus.TGT.embeddings, scale=0.03)

        # Optimizer definition
        optimizer = getattr(torch.optim, config.optimizer)(
            model.parameters(), lr=config.learning_rate)

        date_str, time_str = datetime.now().strftime("%Y%m%d-%H%M%S").split("-")
        result_str = "{}-{}".format(model_name, time_str)
        config.save_dir = os.path.join(config.save_dir, date_str, result_str)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        # logger
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "session.log"))
        logger.addHandler(fh)

        # save config
        params_file = os.path.join(config.save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Saved params to '{}'".format(params_file))

        logger.info(model)

        # Train
        logger.info("Training starts ...")
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          train_iter=train_iter,
                          valid_iter=valid_iter,
                          logger=logger,
                          valid_metric_name="-loss",
                          num_epochs=config.num_epochs,
                          save_dir=config.save_dir,
                          log_steps=config.log_steps,
                          valid_steps=config.valid_steps,
                          grad_clip=config.grad_clip,
                          save_summary=True)
        trainer.train()
        logger.info("Training done!")

        # Test
        logger.info("")
        trainer.load(os.path.join(config.save_dir, "best"))

        logger.info("Testing starts ...")
        logger.info(evaluate(model, test_iter).report_cum())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
