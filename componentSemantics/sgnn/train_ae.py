import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GAE, VGAE

from sgnn.dataset import DependencyCommunityDataset
from sgnn.models import Encoder

dataset = DependencyCommunityDataset("../community")

channels = 8
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(dataset.num_features)

mode = "VGAE"
model = VGAE(Encoder(dataset.num_features, channels, mode=mode)).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(1, 2000):
    model.train()
    for data in loader:
        data.to(dev)
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        print(loss.sum().item())

        if model.encoder.mode in ['VGAE']:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

#    if epoch % 10 == 0:
#        model.eval()
#        with torch.no_grad():
#            z = model.encode(x, train_pos_edge_index)
#        return model.test(z, pos_edge_index, neg_edge_index)

# %%
