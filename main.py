import torch
from torchvision import datasets, transforms
from torchvision.models import vgg11
from torch.utils.data import DataLoader

from aggregator import Aggregator
from frameworks import FedClient, FedServer
from models import SimpleCnn
from partitioner import DataPartitioner
from privacy import VanillaDp
from trainer import Trainer
from torch import nn

device = 'cuda'

if __name__ == "__main__":
    n_client = 10
    n_round = 50
    n_epoch = 5
    privacy_budget = 1
    privacy_relax = 0.1
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    clip_thr = {
        'conv1.weight': 2,
        'conv1.bias': 0.3,
        'conv2.weight': 2,
        'conv2.bias': 0.3,
        'fc1.weight': 2,
        'fc1.bias': 0.4,
        'fc2.weight': 2,
        'fc2.bias': 0.4,
        'fc3.weight': 2,
        'fc3.bias': 0.3,
    }

    testloader = DataLoader(testset, batch_size=64)

    partition = DataPartitioner(
        dataset,
        'dirichlet',
        n_client,
        {'show': True, 'alpha': 0.5, 'batch_size': 64}
    )

    local_models = [SimpleCnn().to(device) for _ in range(n_client)]
    global_model = SimpleCnn().to(device)


    clients = [FedClient(
        local_models[i],
        Trainer(
            local_models[i],
            partition[i],
            nn.CrossEntropyLoss().to(device),
            {'lr': 0.05}
        ),
        {'epoch': n_epoch, 'test': False},
        testloader,
        VanillaDp({
            'n_aggr': n_round,
            'n_dataset': len(partition[i].dataset),
            'epsilon': privacy_budget,
            'delta': privacy_relax,
            'clip_thr': clip_thr,
        })
    ) for i in range(n_client)]

    server = FedServer(
        global_model,
        Aggregator(global_model),
        partition.freq,
        {},
        testloader
    )

    for r in range(n_round):
        weights = {}
        for client in clients:
            client.train()
            weights[client.id] = client.get_weights()

        server.aggregate(weights)
        server.log_test(r)
        glob_weight = server.model.state_dict()



        for client in clients:
            client.set_weights(glob_weight)


