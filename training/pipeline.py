import os
import sys
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM
from pathlib import Path
import wget

sys.path.insert(0, os.path.abspath(Path(__file__).parents[1].resolve()))
from training.src.datasets.opus_dataset import OpusDataset
from training.src.base_model import BaseModel
from constants import OPUS_LINKS, OPUS_PATHS

parser = ArgumentParser(
    prog="seq2seq trainer",
    description="pipeline to train seq2seq models",
    formatter_class=ArgumentDefaultsHelpFormatter,
    )

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

args = parser.parse_args()
stage = args.stage
model_name = args.model
config = args.path_yaml
dataset = args.dataset
dataset_path = args.dataset_path

if stage == "download":
    if dataset.lower() == "opus":
        for ds_name in OPUS_LINKS:
            url = OPUS_LINKS[ds_name]
            wget.download(url, out=dataset_path)
if stage == "preprocess":
    # "no preprocessing implemented yet"
    pass

if dataset.lower() == "opus":
    train = OpusDataset(os.path.join(dataset_path, OPUS_PATHS["train_en"]), 
                        os.path.join(dataset_path, OPUS_PATHS["train_ru"]))
    val = OpusDataset(os.path.join(dataset_path, OPUS_PATHS["dev_en"]), 
                        os.path.join(dataset_path, OPUS_PATHS["dev_ru"]))
    test = OpusDataset(os.path.join(dataset_path, OPUS_PATHS["test_en"]), 
                        os.path.join(dataset_path, OPUS_PATHS["test_ru"]))

net = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
with open(config, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
model = BaseModel(net, model_name, cfg)
model.fit(train, val)
