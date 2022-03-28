import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 75),
            nn.ReLU(),
            nn.Linear(75, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 5000)
        )

    def forward(self, x):
        return self.model(x)
