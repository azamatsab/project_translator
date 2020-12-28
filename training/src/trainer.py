"""Model class, to be extended by specific types of models."""
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

import yaml
import mlflow
import torch
from torch import device
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


sys.path.insert(0, os.path.abspath(Path(__file__).parents[2].resolve()))
from training.src.datasets.opus_dataset import OpusDataset
from training.src.utils import get_configs

cfg = get_configs()

with open(cfg["PATH_LOGGING_CONF"]) as config_fin:
    logging.config.dictConfig(yaml.safe_load(config_fin))

class Trainer:
    def __init__(self, model, config, device=0):
        self.model = model
        self.net = self.model.model
        self.device_name = config["net"]["device"]
        
        if config["net"]["ddp"]:
            self.device = device
            self.model.set_mode("ddp", device)
        else:
            self.device = torch.device(self.device_name)
        
        self.model.to(self.device)
        self.config = config
        self.optimizer = self.model.optimizer(config["net"]["lr"])
        self.scheduler = self.model.scheduler()
        logging.info(f"Trainer initialized with {model.name}, on device: {self.device_name}")

    def iteration(self, loader, train=True):
        t_loss = 0
        perplexity = 0
        num_steps = 0
        for batch in tqdm(loader):
            if train:
                self.optimizer.zero_grad()
            loss = self.model.training_step(batch, self.device)
            if train:
                loss.backward()
                self.optimizer.step()
            t_loss += loss.item()
            perplexity += torch.exp(loss).item()
            num_steps += 1

        t_loss /= num_steps
        perplexity /= num_steps
        return t_loss, perplexity

    def run_one_epoch(self, loader, train=True):
        if train:
            self.model.train()
            t_loss, perplexity = self.iteration(loader, train)
        else:
            self.model.eval()
            with torch.no_grad():
                t_loss, perplexity = self.iteration(loader, train)

        return t_loss, perplexity

    def create_loaders(self, train_dataset, val_dataset, world_size):
        num_workers = self.config["loader"]["num_workers"]
        batch_size = self.config["net"]["batch_size"]

        tr_sampler = None
        val_sampler = None

        if self.config["net"]["ddp"]:
            tr_sampler = DistributedSampler(
                train_dataset, rank=self.device, num_replicas=world_size, shuffle=True
            )

            val_sampler = DistributedSampler(
                val_dataset, rank=self.device, num_replicas=world_size, shuffle=False
            )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=tr_sampler, 
                          num_workers=num_workers, pin_memory=True, drop_last=True)
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, 
                          num_workers=num_workers, pin_memory=True, drop_last=False)
        else:
            logging.warning("Validation dataset wasnt passed. Do not rely on training dataset results!")

        return train_loader, val_loader
        

    def fit(self, train_dataset, val_dataset=None, world_size=None):
        logging.info(f"Start fit, train dataset length {len(train_dataset)}")
        experiment_id = mlflow.set_experiment(self.model.name)
        num_workers = self.config["loader"]["num_workers"]
        batch_size = self.config["net"]["batch_size"]
        epochs = self.config["net"]["epoch"]
        lr = self.config["net"]["lr"]
        logging.info(f"Parameters: batch_size: {batch_size}, epochs: {epochs}, lr: {lr}")

        train_loader, val_loader = self.create_loaders(train_dataset, val_dataset, world_size)
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("lr", lr)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("optimizer", self.optimizer)
            mlflow.log_param("scheduler", self.scheduler)
            
            for epoch in range(epochs):
                t_loss, perplexity = self.run_one_epoch(train_loader, train=True)                
                mlflow.log_metric("train loss", t_loss, step=epoch)
                mlflow.log_metric("train perplexity", perplexity, step=epoch)
                logging.info(f"train: Epoch: {epoch + 1}, loss: {round(t_loss, 4)}, perplexity: {round(perplexity)}")
                if val_dataset:
                    val_loss, val_perplexity = self.run_one_epoch(val_loader, train=False)
                    mlflow.log_metric("val loss", val_loss, step=epoch)
                    mlflow.log_metric("val perplexity", val_perplexity, step=epoch)    
                    logging.info(f"val: Epoch: {epoch + 1}, loss: {round(val_loss, 4)}, perplexity: {round(val_perplexity)}")
                    
                    path_to_save = self.model.weights_filename + f"_{epoch + 1}_{val_loss}"
                    self.model.save_model(path_to_save)
                    logging.debug(f"Saving model {path_to_save}")
                    if (epoch + 1) % self.config["net"]["calculate_bleu_step"] == 0:
                        metric = self.calculate_metrics(val_dataset)
                        mlflow.log_metric(self.model.metrics()[0], metric, step=epoch)
            if self.config["net"]["use_scheduler"]:
                self.scheduler.step()

    def calculate_metrics(self, dataset, world_size=None):
        logging.info(f"Start calculating {self.model.metrics()[0]} metric")
        num_workers = self.config["loader"]["num_workers"]
        batch_size = self.config["net"]["batch_size"]

        _, loader = self.create_loaders(dataset, dataset, world_size)

        metric = 0
        samples = 0
        for batch in tqdm(loader):
            metric += self.model.calculate_metrics(batch, self.device)
            samples += len(batch["source"])
        metric_mean_val = round(metric / samples, 2)
        logging.info(f"{self.model.metrics()[0]}: {metric_mean_val}")
        return metric_mean_val