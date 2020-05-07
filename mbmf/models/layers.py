import torch
import torch.nn as nn

class EnsembleLinearLayer(nn.Module):
    def __init__(self, in_size, out_size, ensemble_size, init_type="xavier_normal"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.ensemble_size = ensemble_size
        self.init_type = init_type
        self.reset_parameters()

    def forward(self, x):
        return torch.baddbmm(self.biases, x, self.weights)

    def reset_parameters(self):
        weights = torch.zeros(self.ensemble_size, self.in_size, self.out_size).float()
        biases = torch.zeros(self.ensemble_size, 1, self.out_size).float()

        for weight in weights:
            self._init_weight(weight, self.init_type)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def _init_weight(self, weight, init_type):
        if init_type == "xavier_uniform":
            nn.init.xavier_uniform_(weight)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(weight)
