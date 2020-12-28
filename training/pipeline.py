import os
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import wget
import yaml
from transformers import AutoModelForSeq2SeqLM

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.abspath(Path(__file__).parents[1].resolve()))
from training.src.datasets.opus_dataset import OpusDataset
from training.src.trainer import Trainer
from training.src.setup_ddp import setup, cleanup
from translator.src.seq2seq import Seq2SeqModel

DEFAULT_CONFIG_FILEPATH = os.sep.join(
    [
        os.path.dirname(__file__),
        'constants.yml',
    ]
)

with open(DEFAULT_CONFIG_FILEPATH, 'r') as fin:
    cfg = yaml.safe_load(fin)

with open(cfg["PATH_LOGGING_CONF"]) as config_fin:
    logging.config.dictConfig(yaml.safe_load(config_fin))

def setup_parser(parser):
    parser.add_argument("--stage", default="dataset",
        help="possible values: download, preprocess, dataset", dest="stage")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-en-ru",
        help="choose a model name", dest="model")
    parser.add_argument("--dataset", default="OPUS",
        help="dataset name, possible values: OPUS, opus)", dest="dataset")
    parser.add_argument("--dataset_path", default="dataset",
        help="path to save dataset", dest="dataset_path")
    parser.add_argument("--path_to_yaml_params", default="training/config.yaml",
        help="path to load configs", dest="path_yaml")
    return parser

def handle_stages(args):
    logging.info(f"Starting from stage '{args.stage}'")
    if args.stage == "download":
        if dataset.lower() == "opus":
            for ds_name in cfg["OPUS_LINKS"]:
                url = cfg["OPUS_LINKS"][ds_name]
                wget.download(url, out=dataset_path)
    if args.stage == "preprocess":
        logging.warning("Preprocessing did not implemented yet")

def load_datasets(args):
    logging.info("Loading datasets")
    dataset = args.dataset
    dataset_path = args.dataset_path
    if dataset.lower() == "opus":
        train = OpusDataset(os.path.join(dataset_path, "opus.en-ru-train.en"), 
                            os.path.join(dataset_path, "opus.en-ru-train.ru"))
        val = OpusDataset(os.path.join(dataset_path, "opus.en-ru-dev.en"), 
                            os.path.join(dataset_path, "opus.en-ru-dev.ru"))
        test = OpusDataset(os.path.join(dataset_path, "opus.en-ru-test.en"), 
                            os.path.join(dataset_path, "opus.en-ru-test.ru"))
        return train, val, test
    else:
        logging.warning(f"Dataset {dataset} not defined yet")

def run_train(rank, world_size, model, train, val, test, cfg):
    setup(rank, world_size)
    trainer = Trainer(model, cfg, rank)
    trainer.fit(train, val, world_size)
    trainer.calculate_metrics(test)
    cleanup()

def run_multigpu(fn, world_size):
    mp.spawn(fn,
             args=(world_size, model, train, val, test, cfg),
             nprocs=world_size,
             join=True)

def run(model, train, val, test, cfg):
    logging.info("Start running training pipeline")
    if cfg["net"]["ddp"]:
        run_multigpu(run_train, torch.cuda.device_count())
    else:
        trainer = Trainer(model, cfg)
        trainer.fit(train, val)
        trainer.calculate_metrics(test)

if __name__ == '__main__':
    parser = ArgumentParser(
        prog="seq2seq trainer",
        description="pipeline to train seq2seq models",
        formatter_class=ArgumentDefaultsHelpFormatter,
        )

    parser = setup_parser(parser)
    args = parser.parse_args()
    handle_stages(args)
    train, val, test = load_datasets(args)    
    model_name = args.model
    
    with open(args.path_yaml, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    model = Seq2SeqModel(model_name)
    run(model, train, val, test, cfg)
