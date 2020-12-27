import os
import sys
import yaml
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM
from pathlib import Path
import wget

sys.path.insert(0, os.path.abspath(Path(__file__).parents[1].resolve()))
from training.src.datasets.opus_dataset import OpusDataset
from training.src.base_model import BaseModel
from constants import OPUS_LINKS, OPUS_PATHS

logger = logging.getLogger(__name__)


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

def handle_stages(args):
    if args.stage == "download":
        logger.info("Downloading dataset")
        if dataset.lower() == "opus":
            for ds_name in OPUS_LINKS:
                url = OPUS_LINKS[ds_name]
                wget.download(url, out=dataset_path)
    if args.stage == "preprocess":
        logger.warning("Preprocessing did not implemented yet")

def load_datasets(args):
    logger.info("Loading datasets")
    dataset = args.dataset
    dataset_path = args.dataset_path
    if dataset.lower() == "opus":
        train = OpusDataset(os.path.join(dataset_path, OPUS_PATHS["train_en"]), 
                            os.path.join(dataset_path, OPUS_PATHS["train_ru"]))
        val = OpusDataset(os.path.join(dataset_path, OPUS_PATHS["dev_en"]), 
                            os.path.join(dataset_path, OPUS_PATHS["dev_ru"]))
        test = OpusDataset(os.path.join(dataset_path, OPUS_PATHS["test_en"]), 
                            os.path.join(dataset_path, OPUS_PATHS["test_ru"]))
        return train, val, test
    else:
        logger.warning(f"Dataset {dataset} not defined yet")

def run(model, train, val, test, cfg):
    logger.info("Start running training pipeline")
    trainer = Trainer(model, cfg)
    val = OpusDataset(os.path.join(DATASET_PATH, TEST_FILE_EN), 
                      os.path.join(DATASET_PATH, TEST_FILE_RU))
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
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # run(model, train, val, test, cfg)
