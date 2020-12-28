import os

import pytest
import yaml

from translator.src.seq2seq import Seq2SeqModel
from training.src.datasets.opus_dataset import OpusDataset
from training.pipeline import run

MODEL_NAME = 'Helsinki-NLP/opus-mt-en-ru'
MODEL_NAME2 = 't5-small'
CONFIG_PATH = 'training/config.yaml'
DATASET_PATH = 'dataset'
TEST_FILE_EN = 'test_en'
TEST_FILE_RU = 'test_ru'


def test_run_without_error():
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    val = OpusDataset(os.path.join(DATASET_PATH, TEST_FILE_EN), 
                      os.path.join(DATASET_PATH, TEST_FILE_RU))

    model = Seq2SeqModel(MODEL_NAME)
    run(model, val, val, val, cfg)