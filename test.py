import torch
from autoencoder import Autoencoder
from ecg_dataset import EcgDataset
import plotly.graph_objs as go
import random

autoenc = Autoencoder()
autoenc.load_state_dict(torch.load("./trained-model-state-dict.pt"))
autoenc.eval()
class DROP:
    def __init__(self, v):
        self.v = v

    def __call__(self, oneDTensor):
        for i in range(oneDTensor.shape[0]):
            if random.random() <= self.v:
                oneDTensor[i] = 0
        return

v = 0.1
drop = DROP(v)
ds_test = EcgDataset(None, slice=slice(14000, None))

lead = 0

randI = list(range(len(ds_test)))
random.shuffle(randI)
with torch.no_grad():
    for i in randI:
        ds_test.view_encounter(i, lead_id=lead)

        signal = ds_test[i][0][lead]
        drop(signal)
        reconstructed = autoenc(signal)
        fig = go.Figure(go.Scatter(y=reconstructed.detach().numpy(), mode="markers", marker=dict(color="red")))
        fig.show()
        print(((signal - reconstructed) * (signal - reconstructed)).sum().item())
        print()


