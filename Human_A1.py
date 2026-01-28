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
file_fold = '..../data/Human_Lymph_Nodes/A1/'
omics2_type = 'ADT'
random_seed = 2024
fix_seed(random_seed)

# ========== 加载数据 ==========
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
# print(adata_omics1)
# print(adata_omics2)
# ========== 数据预处理 ==========
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)
adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)

adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)


data = construct_neighbor_graph(adata_omics1, adata_omics2, n_neighbors=4, n_neighbors_gene=17)

# ========== 超参数设置 ==========
best_params = {
    'dim_input':64,
    'dim_output': 128,
    'epochs':600,
    'weight_factor1': 1,
    'weight_factor2': 1,
    'weight_factor3': 5,
    'weight_factor4': 5,
    'alpha': 0.4,
    'contrastive_temperature': 0.36,
    'graphcl_temperature': 0.1,
    'drop_edge_prob': 0.3
}

# ========== 训练模型 ==========
model = train_MAGCL(data,
                 device=device,
                 epochs=best_params['epochs'],
                 dim_input=best_params['dim_input'],
                 dim_output=best_params['dim_output'],
                 weight_factors=[best_params['weight_factor1'],
                                 best_params['weight_factor2'],
                                 best_params['weight_factor3'],
                                 best_params['weight_factor4'],
                                 ],
                 alpha=best_params['alpha'],
                use_contrastive=True,
                use_graphcl=True,
                contrastive_temperature=best_params['contrastive_temperature'],
                graphcl_temperature=best_params['graphcl_temperature'],
                drop_edge_prob=best_params['drop_edge_prob'],
                omics2_type=omics2_type)

output = model.train()

# ========== 保存表示 ==========
adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1']
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2']
adata.obsm['MAGCL'] = output['MAGCL']

# ========== 聚类与可视化 ==========
clustering(adata, key='MAGCL', add_key='MAGCL', n_clusters=10, method='mclust', use_pca=True)
label = adata.obs['MAGCL']
# 提取 ground_truth 和 聚类 列
ground_truth = adata.obs['Spatial_Label']
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

# # 可视化
# fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
# sc.pp.neighbors(adata, use_rep='MAGCL', n_neighbors=10)
# sc.tl.umap(adata)
# sc.pl.umap(adata, color='MAGCL', ax=ax_list[0], title='MAGCL', s=20, show=False)
# sc.pl.embedding(adata, basis='spatial', color='MAGCL', ax=ax_list[1], title= f'MAGCL-ARI:{adjusted_rand_score(ground_truth, label):.4f}', s=25, show=False)
# plt.tight_layout(w_pad=0.3)
# plt.show()


# ========== 保存实验记录 ==========
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("/..../", exist_ok=True)
log_record = {
    "timestamp": timestamp,
    "params": best_params,
    "metrics": metrics
}
with open(f"..../exp_{timestamp}.json", 'w') as f:
    json.dump(log_record, f, indent=4)
print(f" 实验记录已保存至 ..../exp_{timestamp}.json")


