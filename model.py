
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, GINConv

class SAGEEncoder(nn.Module):
    def __init__(self, in_feat, out_feat, dropout_rate=0.5, activation=F.relu):
        super().__init__()
        self.sage = SAGEConv(in_feat, out_feat)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, feat, adj):
        edge_index = adj.coalesce().indices()
        x = self.dropout(feat)
        x = self.sage(x, edge_index)
        x = self.activation(x)
        return x

class SAGEDecoder(SAGEEncoder):
    pass

class GATv2Encoder(nn.Module):
    def __init__(self, in_feat, out_feat, dropout_rate=0.5, heads=2, activation=F.elu):
        super().__init__()
        self.gat = GATv2Conv(in_channels=in_feat, out_channels=out_feat, heads=heads, concat=False, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, feat, adj):
        edge_index = adj.coalesce().indices()
        x = self.dropout(feat)
        x = self.gat(x, edge_index)
        x = self.activation(x)
        return x

class GATv2Decoder(GATv2Encoder):
    pass

class GINEncoder(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.ReLU(),
            nn.Linear(out_feat, out_feat)
        )
        self.gin = GINConv(mlp)

    def forward(self, feat, adj):
        edge_index = adj.coalesce().indices()
        return self.gin(feat, edge_index)

class GINDecoder(GINEncoder):
    pass

class MGAFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MGAFusion, self).__init__()
        self.gate_rna = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        self.gate_other = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        self.fc_layer = nn.Linear(input_dim * 2, 2)
        self.fusion_layer = nn.Linear(input_dim, output_dim)

    def forward(self, z1, z2):
        z_stacked = torch.cat([z1, z2], dim=1)
        z1 = z1 * self.gate_rna(z1)
        z2 = z2 * self.gate_other(z2)
        attn = F.softmax(self.fc_layer(z_stacked), dim=1)
        z_fused = attn[:, 0:1] * z1 + attn[:, 1:2] * z2
        return F.normalize(self.fusion_layer(z_fused), p=2, dim=1)

class Encoder_overall(nn.Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1,
                 dim_in_feat_omics2, dim_out_feat_omics2,
                 dropout=0.0, act=F.relu, omics2_type='ADT'):
        super().__init__()
        self.encoder_omics1 = SAGEEncoder(dim_in_feat_omics1, dim_out_feat_omics1)
        self.decoder_omics1 = SAGEDecoder(dim_out_feat_omics1, dim_in_feat_omics1)

        if omics2_type == 'ADT':
            self.encoder_omics2 = GATv2Encoder(dim_in_feat_omics2, dim_out_feat_omics2)
            self.decoder_omics2 = GATv2Decoder(dim_out_feat_omics2, dim_in_feat_omics2)
        elif omics2_type == 'ATAC':
            self.encoder_omics2 = GINEncoder(dim_in_feat_omics2, dim_out_feat_omics2)
            self.decoder_omics2 = GINDecoder(dim_out_feat_omics2, dim_in_feat_omics2)
        else:
            raise ValueError("omics2_type must be 'ADT' or 'ATAC'")

        self.fusion = MGAFusion(dim_out_feat_omics1, dim_out_feat_omics2)

    def forward(self, features_omics1, features_omics2, adj1_omics1, adj2_omics2):
        z1 = self.encoder_omics1(features_omics1, adj1_omics1)
        z2 = self.encoder_omics2(features_omics2, adj2_omics2)
        z_combined = self.fusion(z1, z2)
        r1 = self.decoder_omics1(z_combined, adj1_omics1)
        r2 = self.decoder_omics2(z_combined, adj2_omics2)
        return {
            'emb_latent_omics1': z1,
            'emb_latent_omics2': z2,
            'emb_latent_combined': z_combined,
            'emb_recon_omics1': r1,
            'emb_recon_omics2': r2
        }
