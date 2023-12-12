from torchvision import datasets, transforms
import numpy as np
from partitioner import DataPartitioner

def iid_partition(dataset, n_client):
    indices = np.array([i for i in range(len(dataset))])
    np.random.shuffle(indices)
    indices = np.split(indices, n_client)

    return {i: indices[i] for i in range(n_client)}

if __name__ == "__main__":
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
    )

    partition = DataPartitioner(
        dataset,
        'dirichlet',
        10,
        {'show': True, 'alpha': 0.1, 'batch_size': 64}
    )

    print(partition[0])