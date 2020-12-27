import os
import pytest
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from translator.base_model import BaseModel
from training.src.trainer import Trainer
from training.src.datasets.opus_dataset import OpusDataset
from training.src.metrics import calculate_bleu
from training.constants import WEIGHTS_DIR

MODEL_NAME = 'Helsinki-NLP/opus-mt-en-ru'
MODEL_NAME2 = 't5-small'
CONFIG_PATH = 'training/config.yaml'
DATASET_PATH = 'dataset'
TEST_FILE_EN = 'test_en'
TEST_FILE_RU = 'test_ru'


class TestModel(BaseModel):
    def __init__(self, tokenizer_filepath: str):
        # logger.info("Initialize %s class with pretrained model %s", self.__class__.__name__, tokenizer_filepath)
        self.name = f'{self.__class__.__name__}'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_filepath)

    @property
    def weights_filename(self) -> str:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTS_DIR / f'{self.name}_weights.h5')

    def training_step(self, batch, device):
        src_str = batch["source"]
        target_str = batch["target"]
        inputs = self.tokenizer.prepare_seq2seq_batch(src_str, target_str, return_tensors="pt")  # "pt" for pytorch
        input_ids = inputs.input_ids.to(device, dtype=torch.long)
        attention_mask = inputs.attention_mask.to(device, dtype=torch.long)
        labels = inputs.attention_mask.to(device, dtype=torch.long)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss

    def evaluate(self, batch, device):
        with torch.no_grad():
            loss = training_step(batch, device)
        return loss

    def metrics(self):
        return ["bleu"]

    def calculate_metrics(self, batch, device):
        src_str = batch["source"]
        target_str = batch["target"]
        inputs = self.tokenizer.prepare_seq2seq_batch(src_str, target_str, return_tensors="pt")  # "pt" for pytorch
        input_ids = inputs.input_ids.to(device, dtype=torch.long)
        attention_mask = inputs.attention_mask.to(device, dtype=torch.long)

        translated = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        tgt_text = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        bleu = calculate_bleu(tgt_text, target_str)["bleu"]
        return bleu

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def optimizer(self, lr, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, **kwargs)
        return self.optimizer

    def scheduler(self):
        return get_constant_schedule(self.optimizer)

    def save_model(self, filepath: str):
        """ Save the trained model
        """
        # logger.info('Store the model to path: "%s"', filepath)
        self.tokenizer.save_pretrained(filepath)
        self.model.save_pretrained(filepath)
        
    def load_model(self, filepath: str):
        """ Load the trained model
        """
        # logger.info('Load the model for path: "%s"', filepath)
        self.tokenizer.from_pretrained(filepath)
        self.model.from_pretrained(filepath)


@pytest.fixture
def get_model():
    return TestModel(MODEL_NAME)

@pytest.fixture
def get_model2():
    return TestModel(MODEL_NAME)

def test_config_file():
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    assert 'net' in cfg
    assert 'loader' in cfg
    assert 'batch_size' in cfg['net']
    assert 'num_workers' in cfg['loader']

def test_fit(get_model):
    net = get_model
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg["net"]["epoch"] = 2
    cfg["net"]["batch_size"] = 2
    cfg["net"]["device"] = "cpu"
    trainer = Trainer(net, cfg)

    val = OpusDataset(os.path.join(DATASET_PATH, TEST_FILE_EN), 
                      os.path.join(DATASET_PATH, TEST_FILE_RU))
    trainer.fit(val, val)

# def test_fit2(get_model2):
#     net = get_model2
#     with open(CONFIG_PATH, "r") as ymlfile:
#         cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
#     cfg["net"]["epoch"] = 2
#     cfg["net"]["batch_size"] = 2
#     trainer = Trainer(net, cfg)

#     val = OpusDataset(os.path.join(DATASET_PATH, TEST_FILE_EN), 
#                       os.path.join(DATASET_PATH, TEST_FILE_RU))
#     trainer.fit(val, val)

def test_calculate_metrics(get_model):
    net = get_model
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg["net"]["epoch"] = 2
    cfg["net"]["batch_size"] = 2
    trainer = Trainer(net, cfg)
    val = OpusDataset(os.path.join(DATASET_PATH, TEST_FILE_EN), 
                      os.path.join(DATASET_PATH, TEST_FILE_RU))
    trainer.calculate_metrics(val)


def test_bleu():
    line1 = ["привет"]
    line2 = ["привут"]

    assert calculate_bleu(line1, line2) != 0