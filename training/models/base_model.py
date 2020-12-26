"""Model class, to be extended by specific types of models."""
import yaml
from pathlib import Path

import mlflow
import torch
from torch import device
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
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


DIRNAME = Path(__file__).parents[1].resolve() / 'weights'

conf_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    }


class BaseModel:
    def __init__(self, model, tokenizer_name, config):
        self.name = f'{self.__class__.__name__}_{tokenizer_name}'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model
        self.device = torch.device(config["net"]["device"])
        self.model.to(self.device)
        self.config = config
        self.optimizer = None

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.pth')

    def _get_lr_scheduler(self, num_training_steps):
        schedule_func = conf_to_scheduler[self.config["net"]["scheduler"]]
        if self.config["net"]["scheduler"] == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.config["net"]["scheduler"] == "constant_w_warmup":
            scheduler = schedule_func(self.optimizer, num_warmup_steps=self.config["net"]["warmup_steps"])
        else:
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.config["net"]["warmup_steps"], num_training_steps=num_training_steps
            )
        return scheduler

    def fit(self, dataset):
        num_workers = self.config["loader"]["num_workers"]
        batch_size = self.config["net"]["batch_size"]
        epochs = self.config["net"]["epoch"]
        lr = self.config["net"]["lr"]

        if self.config["net"]["optimizer"] == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        if self.config["net"]["optimizer"] == "Adafactor":
            self.optimizer = Adafactor(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.scheduler = self._get_lr_scheduler(self.config["net"]["epoch"])

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True, drop_last=True)

        for epoch in range(epochs):
            for batch in train_loader:
                src_str = [item[0] for item in batch]
                target_str = [item[1] for item in batch]
                inputs = self.tokenizer.prepare_seq2seq_batch(src_str, target_str, return_tensors="pt")  # "pt" for pytorch
                out = self.model(**inputs)
                self.optimizer.zero_grad()
                loss = out.loss
                loss.backward()
            self.scheduler.step()

    def evaluate():
        pass
