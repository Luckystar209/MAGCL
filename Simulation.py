# === Simulation实验主流程完整代码（含模型训练、评估与保存）===
import os
import json
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from datetime import datetime
from preprocess import fix_seed, clr_normalize_each_cell, construct_neighbor_graph, pca
from train_model import train_MAGCL
from utils import clustering
from metric import jaccard, F_measure
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    v_measure_score, homogeneity_score, completeness_score,
    fowlkes_mallows_score, mutual_info_score
)


# ========== 设置 ==========
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'D:\\R-4.4.1'
file_fold = '.../data/simulation/'
omics2_type = 'ADT'
random_seed = 2024
fix_seed(random_seed)

# ========== 加载数据 ==========
adata_omics1 = sc.read_h5ad(file_fold + 'simulated_data_rna.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'simulated_data_adt.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# ========== 数据预处理 ==========
# RNA
sc.pp.filter_genes(adata_omics1, min_cells=50)
sc.pp.filter_genes(adata_omics1, min_counts=10)
sc.pp.normalize_total(adata_omics1, target_sum=1e6)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=2000)
sc.pp.scale(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)

# Protein
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)

# ========== 构建邻接图 ==========
data = construct_neighbor_graph(adata_omics1, adata_omics2, n_neighbors=6, n_neighbors_gene=18)

# ========== 超参数设置 ==========
best_params = {
    'learning_rate': 0.0008,
    'weight_decay': 0.0001,
    'epochs': 400,
    'dim_output': 112,
    'weight_factor1': 1.66,
    'weight_factor2': 5.41,
    'weight_factor3': 1.59,
    'weight_factor4': 1.98,
    'alpha': 0.58
}

# ========== 训练模型 ==========
model = train_MAGCL(data,
                 device=device,
                 learning_rate=best_params['learning_rate'],
                 weight_decay=best_params['weight_decay'],
                 epochs=best_params['epochs'],
                 dim_output=best_params['dim_output'],
                 weight_factors=[best_params['weight_factor1'],
                                 best_params['weight_factor2'],
                                 best_params['weight_factor3'],
                                 best_params['weight_factor4']
                                 ],
                 alpha=best_params['alpha'])

output = model.train()

# ========== 保存表示 ==========
adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1']
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2']
adata.obsm['MAGCL'] = output['MAGCL']

# ========== 聚类与可视化 ==========
clustering(adata, key='MAGCL', add_key='MAGCL', n_clusters=5, method='mclust', use_pca=True)

# 提取 ground_truth 和 聚类 列
ground_truth = adata.obs['ground_truth']
label = adata.obs['MAGCL']

# 计算 ARI
ari_score = adjusted_rand_score(ground_truth, label)
metrics = {
    "Jaccard": jaccard(label, ground_truth),
    "F_measure": F_measure(label, ground_truth),
    "Mutual_Info": mutual_info_score(ground_truth, label),
    "NMI": normalized_mutual_info_score(ground_truth, label),
    "AMI": adjusted_mutual_info_score(ground_truth, label),
    "V_measure": v_measure_score(ground_truth, label),
    "Homogeneity": homogeneity_score(ground_truth, label),
    "Completeness": completeness_score(ground_truth, label),
    "ARI": adjusted_rand_score(ground_truth, label),
    "FMI": fowlkes_mallows_score(ground_truth, label)
}

for k, v in metrics.items():
    print(f"{k}: {v:.6f}")

# 可视化
# fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
# sc.pp.neighbors(adata, use_rep='MGACL', n_neighbors=10)
# sc.tl.umap(adata)
# sc.pl.umap(adata, color='MGACL', ax=ax_list[0], title='MGACL', s=50, show=False)
# sc.pl.embedding(adata, basis='spatial', color='MGACL', ax=ax_list[1], title=f'ARI:{ari_score:.4f}', s=80, show=False)
# plt.tight_layout(w_pad=0.3)
# plt.show()



