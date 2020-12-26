import os
import pytest
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer

from training.models.base_model import BaseModel
from training.datasets.opus_dataset import OpusDataset

MODEL_NAME = 'Helsinki-NLP/opus-mt-en-ru'
CONFIG_PATH = 'training/config.yaml'
DATASET_PATH = 'dataset'
TEST_FILE_EN = 'test_en'
TEST_FILE_RU = 'test_ru'

@pytest.fixture
def get_model():
    return MarianMTModel.from_pretrained(MODEL_NAME)

def test_config_file():
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    assert 'net' in cfg
    assert 'loader' in cfg
    assert 'batch_size' in cfg['net']
    assert 'num_workers' in cfg['loader']

def test_base_model_has_name(get_model):
    net = get_model
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    model = BaseModel(net, MODEL_NAME, cfg)
    assert isinstance(model.name, str)
    assert len(model.name) > 0

def test_base_model_weights_named_by_model(get_model):
    net = get_model
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    model = BaseModel(net, MODEL_NAME, cfg)
    weights_dir = model.weights_filename
    assert model.name in weights_dir

def test_fit(get_model):
    net = get_model
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg["net"]["epoch"] = 10
    cfg["net"]["batch_size"] = 2
    model = BaseModel(net, MODEL_NAME, cfg)

    val = OpusDataset(os.path.join(DATASET_PATH, TEST_FILE_EN), 
                      os.path.join(DATASET_PATH, TEST_FILE_RU))
    model.fit(val)

def test_schedulers(get_model):
    types = ["linear",
              "cosine",
              "cosine_w_restarts",
              "polynomial",
              "constant",
              "constant_w_warmup"]

    net = get_model
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg["net"]["epoch"] = 1
    model = BaseModel(net, MODEL_NAME, cfg)
    model.optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for sch_type in types:
        model.config["net"]["scheduler"] = sch_type
        scheduler = model._get_lr_scheduler(10)
        scheduler.step()
