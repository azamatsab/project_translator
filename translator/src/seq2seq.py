import os
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

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

from translator.src.base_model import BaseModel
from training.src.metrics import calculate_bleu

DEFAULT_CONFIG_FILEPATH = os.sep.join(
    [
        os.path.dirname(__file__),
        '../../training/constants.yml',
    ]
)

with open(DEFAULT_CONFIG_FILEPATH, 'r') as fin:
    cfg = yaml.safe_load(fin)

class Seq2SeqModel(BaseModel):
    def __init__(self, tokenizer_filepath: str):
        # logger.info("Initialize %s class with pretrained model %s", self.__class__.__name__, tokenizer_filepath)
        self.name = f'{self.__class__.__name__}'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_filepath)
        self.mode = None

    @property
    def weights_filename(self) -> str:
        weights_dir = Path(cfg["WEIGHTS_DIR"])
        weights_dir.mkdir(parents=True, exist_ok=True)
        return os.path.join(cfg["WEIGHTS_DIR"], f'{self.name}')

    def set_mode(self, mode, rank):
        self.mode = mode
        if mode == "DDP":
            self.model = DDP(self.model, device_ids=[rank])

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
