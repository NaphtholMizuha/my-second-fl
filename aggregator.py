from torch import nn

class Aggregator:
    def __init__(self, model: nn.Module):
        self.model = model

    def aggregate(self, weights: dict, freq: dict):
        aggr = {}

        for client, weight in weights.items():
            for key, value in weight.items():
                if aggr.get(key) is None:
                    aggr[key] = value * freq[client]
                else:
                    aggr[key] += value * freq[client]

        self.model.load_state_dict(aggr)
        self.model.eval()