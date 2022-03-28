import torch
from ecg_dataset import EcgDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from autoencoder import Autoencoder
import random

ds_train = EcgDataset(None, slice=slice(None, 14000))

class DROP:
    # A callable for dropout inplace. v is the proportion of the signal to randomly drop to zero
    def __init__(self, v):
        self.v = v

    def __call__(self, oneDTensor):
        for i in range(oneDTensor.shape[0]):
            if random.random() <= self.v:
                oneDTensor[i] = 0
        return


v = 0.1

drop = DROP(v)

batch_size = 5000
lead = 0

autoenc = Autoencoder().to("cuda")
L = nn.MSELoss(reduction='none')
opt = torch.optim.Adam(autoenc.parameters(), lr=1e-3)
ld = DataLoader(ds_train, batch_size=batch_size, num_workers=10, shuffle=True)
for epoch in range(30):
    last_loss = 0
    for X, zero_vec, ids in ld:
        # Xcut are the signals that will be artificially corrupted (dropout)
        Xcut = X[:, lead, :].to("cuda")
        # Y are the untouched signals (reconstruction target)
        Y = X[:, lead, :].clone().to("cuda")
        ids = np.array(ids)
        for i in range(Xcut.shape[0]):
            # Normalize
            mn = torch.min(Xcut[i])
            rng = torch.max(Xcut[i]) - mn
            Xcut[i] -= mn
            Xcut[i] /= rng
            Y[i] -= mn
            Y[i] /= rng
            # Apply zero mask dropout
            drop(Xcut[i])
            #fig = go.Figure(go.Scatter(y=Xcut[i], mode="markers", marker=dict(color="red")))
            #fig.show()
            #fig = go.Figure(go.Scatter(y=Y[i].cpu(), mode="markers", marker=dict(color="red")))
            #fig.show()

        reconstruction = autoenc(Xcut)
        losses = torch.sum(L(reconstruction, Y), dim=1)
        last_loss = losses.sum()
        opt.zero_grad()
        losses.sum().backward()
        opt.step()
    print(last_loss)

torch.save(autoenc.state_dict(), "./trained-model-state-dict.pt")
print("Done training, saved model state dict.")
