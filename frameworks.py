import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from aggregator import Aggregator
from privacy import BaseDp
from trainer import Trainer
import uuid

device = "cuda"

class FedServer:
    def __init__(
        self,
        model: nn.Module,
        aggregator: Aggregator,
        freq: dict,
        args: dict = {},
        testloader: DataLoader = None
    ):
        self.uuid = uuid.uuid1()
        self.model = model
        self.aggregator = aggregator
        self.freq = freq
        self.args = args
        self.testloader = testloader

    def aggregate(self, weights):
        self.aggregator.aggregate(weights, self.freq)

    def log_test(self, round):
        loss, acc = self.test()
        logger.warning(f"Server Round {round} (UUID: {self.uuid})\nloss={loss:.3f}, acc={acc * 100:.2f}%")

    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        n_batches = len(self.testloader)

        loss, acc = 0, 0

        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(device), y.to(device)
                pred = self.model(x)
                loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss = loss / n_batches
        acc = acc / (n_batches * self.testloader.batch_size)
        return loss, acc

class FedClient:
    count = 0
    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        args: dict = {},
        testloader: DataLoader = None,
        protector: BaseDp = None
    ):
        self.uuid = uuid.uuid1()
        self.id = FedClient.count
        FedClient.count += 1
        self.trainer = trainer
        self.args = args
        self.testloader = testloader
        self.model = model
        self.protector = protector



    def train(self):
        for i in range(self.args['epoch']):
            loss = self.trainer.train()
            logger.debug(f"Client {self.id} Epoch {i} : train_loss={loss:.3f}")
            if self.args['test']:
                loss, acc = self.test()
                logger.debug(f"Client {self.id} Epoch {i}: test_loss={loss:.3f}, acc={acc*100:.2f}%")
        # for k, v in self.model.state_dict().items():
        #     print(f"'{k}': {torch.norm(v)}")

    def get_weights(self):
        if self.protector is None:
            return self.model.state_dict()
        else:
            return self.protector(self.model.state_dict())

    def set_weights(self, weights: dict):
        self.model.load_state_dict(weights)

    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        loss, acc = 0, 0

        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(device), y.to(device)
                pred = self.model(x)
                loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= len(self.testloader)
        acc /= len(self.testloader.dataset)
        return loss, acc

    def __del__(self):
        self.count -= 1