# train_sg.py
import torch
import torch.nn.functional as F
from model import Encoder_overall
from preprocess import adjacent_matrix_preprocessing

def drop_edges(adj, drop_prob=0.2):
    edge_index = adj.coalesce().indices()
    num_edges = edge_index.shape[1]
    mask = torch.rand(num_edges) > drop_prob
    new_edge_index = edge_index[:, mask]
    values = adj.coalesce().values()[mask]
    return torch.sparse_coo_tensor(new_edge_index, values, adj.shape).coalesce()

def graphcl_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.shape[0]).to(z1.device)
    return F.cross_entropy(sim, labels)

class train_MAGCL:
    def __init__(self, data,
                 omics2_type='ADT',
                 device=torch.device('cpu'),
                 learning_rate=0.0001,
                 weight_decay=5e-3,
                 epochs=600,
                 dim_output=64,
                 weight_factors=[1, 1, 1, 1],
                 alpha=0.5):
        self.data = data.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.alpha = alpha
        self.omics2_type = omics2_type

        self.adata_omics1 = data['adata_omics1']
        self.adata_omics2 = data['adata_omics2']

        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, alpha)

        self.features_omics1 = torch.tensor(self.adata_omics1.obsm['feat'], dtype=torch.float32).to(device)
        self.features_omics2 = torch.tensor(self.adata_omics2.obsm['feat'], dtype=torch.float32).to(device)
        self.adj_omics1 = self.adj['fused_adj_omics1'].to(device)
        self.adj_omics2 = self.adj['fused_adj_omics2'].to(device)

        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]

        self.model = Encoder_overall(
            self.dim_input1, self.dim_output,
            self.dim_input2, self.dim_output,
            omics2_type=self.omics2_type
        ).to(self.device)

    def train(self):
        torch.manual_seed(2024)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            self.model.train()
            result = self.model(self.features_omics1, self.features_omics2, self.adj_omics1, self.adj_omics2)

            loss_recon_1 = F.mse_loss(self.features_omics1, result['emb_recon_omics1'])
            loss_recon_2 = F.mse_loss(self.features_omics2, result['emb_recon_omics2'])
            loss_cont = self.compute_contrastive_loss(result['emb_latent_omics1'], result['emb_latent_omics2'])

            adj1_aug = drop_edges(self.adj_omics1)
            adj2_aug = drop_edges(self.adj_omics2)
            result_aug = self.model(self.features_omics1, self.features_omics2, adj1_aug, adj2_aug)
            loss_graphcl = graphcl_loss(result['emb_latent_omics1'], result_aug['emb_latent_omics1']) + \
                           graphcl_loss(result['emb_latent_omics2'], result_aug['emb_latent_omics2'])

            total_loss = (self.weight_factors[0] * loss_recon_1 +
                          self.weight_factors[1] * loss_recon_2 +
                          self.weight_factors[2] * loss_cont +
                          self.weight_factors[3]* loss_graphcl)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {total_loss.item():.4f}")

        with torch.no_grad():
            self.model.eval()
            result = self.model(self.features_omics1, self.features_omics2, self.adj_omics1, self.adj_omics2)

        output = {
            'emb_latent_omics1': F.normalize(result['emb_latent_omics1'], p=2).cpu().numpy(),
            'emb_latent_omics2': F.normalize(result['emb_latent_omics2'], p=2).cpu().numpy(),
            'MAGCL': F.normalize(result['emb_latent_combined'], p=2).cpu().numpy()
        }
        self.model_instance = self.model
        return output

    def compute_CCR_loss(self, emb1, emb2):
        sim = torch.mm(F.normalize(emb1, p=2), F.normalize(emb2, p=2).T)
        return ((sim - torch.eye(sim.shape[0], device=sim.device)) ** 2).mean()

    def compute_contrastive_loss(self, emb1, emb2, temperature=0.2):
        logits = torch.mm(F.normalize(emb1, p=2), F.normalize(emb2, p=2).T) / temperature
        labels = torch.arange(logits.shape[0]).to(logits.device)
        return F.cross_entropy(logits, labels)
