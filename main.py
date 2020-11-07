import argparse
import glob
import logging
import os
import time
import pytorch_lightning as pl
from argparse import Namespace
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning_base import BaseTransformer, add_generic_args, generic_train
from Modules import SemEvalTransformer

logger = logging.getLogger(__name__)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    add_generic_args(parser, os.getcwd())
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = SemEvalTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)
    # ------------
    # data
    # ------------
    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    # val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    # test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = SemEvalTransformer(args)

    # ------------
    # training
    # ------------
    trainer = generic_train(model, args)
    # trainer.fit(model, args)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)

    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-epoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        return trainer.test(model)

if __name__ == '__main__':
    cli_main()
