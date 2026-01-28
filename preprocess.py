import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import ot  # Optimal transport for distance calculation
import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph




def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    # 如果 inplace=False，创建 adata 对象的副本
    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def pca(adata, use_reps=None, n_comps=10):
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix

    pca = PCA(n_components=n_comps, random_state=2024)

    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="correlation",
                               include_self=False):
    """Constructing feature neighbor graph according to 特征矩阵"""

    feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)  # adata_omics1.obsm['feat']是[3484×3000]的特征矩阵,feature_graph_omics1是[3484×3484]
    feature_graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2

def construct_neighbor_graph(adata_omics1, adata_omics2, n_neighbors=3,n_neighbors_gene=20):

    ################# spatial graph #################
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2


    ################# feature graph #################


    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2,k=n_neighbors_gene)  # feature_graph_omics1, feature_graph_omics2是[3484×3484]
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

    return data


def normalize_adj(adj):
    """行归一化邻接矩阵"""
    degree = np.sum(adj, axis=1)
    degree[degree == 0] = 1e-8  # 防止除0
    return adj / degree[:, np.newaxis]

def construct_fusion_graph(spatial_adj_dense, feature_adj_dense, alpha=0.5):
    """
    融合图构建：融合空间图和特征图

    参数：
    - spatial_adj_dense: 空间图（二维 numpy array）
    - feature_adj_dense: 特征图（二维 numpy array）
    - alpha: 加权系数，alpha 越大空间信息占比越大

    返回：
    - 融合后的归一化图：PyTorch 稀疏张量形式
    """
    # 加权融合
    fused_adj = alpha * spatial_adj_dense + (1 - alpha) * feature_adj_dense
    # 二值化处理（可选）
    # fused_adj = np.where(fused_adj > 0, 1, 0)
    # 转稀疏 + 归一化
    return preprocess_graph(fused_adj)
def preprocess_graph(adj):     # adj：矩阵和矩阵转置相加经过值大于2的归于1，二维数组形式
    """对输入的邻接矩阵（adj）进行预处理"""

    # 转换压缩矩阵格式
    adj = sp.coo_matrix(adj)  # adj压缩形式 ：col=[],row=[],value=[]
    # 添加自环
    adj_ = adj + sp.eye(adj.shape[0]) # adj_:adj的对角线加1
    # 计算度矩阵
    rowsum = np.array(adj_.sum(1))    # [3484,1]
    # 计算度矩阵的逆平方根
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())   # [3484,3484]
    # 归一化邻接矩阵
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()   # adj_normalized:adj_normalized的压缩形式col=[],row=[],value=[]
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    # 转换为 COO 格式
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # sparse_mx压缩格式：col=[],row=[],data=[]
    # 构建索引张量
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 根据行和列索引构建索引张量[[行标]，[列标]]
    # 构建值张量
    values = torch.from_numpy(sparse_mx.data)  # 根据data索引构建value张量
    # 定义形状
    shape = torch.Size(sparse_mx.shape)  # shape=【3484，3484】
    # 创建 PyTorch 稀疏张量

    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2,alpha):
    """Converting dense adjacent matrix to sparse adjacent matrix"""

    ######################################## construct spatial graph ########################################
    # 获取空间邻接信息:邻接表（DataFrame格式） adj_spatial
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)

    ######################################## prepare graphs for fusion ########################################
    # 将空间邻接图转换为 PyTorch Tensor（稠密格式）
    adj_spatial_omics1 = torch.FloatTensor(adj_spatial_omics1.toarray())
    adj_spatial_omics2 = torch.FloatTensor(adj_spatial_omics2.toarray())


    # 将特征邻接图也转换为 PyTorch Tensor（已是稠密矩阵）
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())


    ######################################## fuse spatial and feature graphs ########################################
    fused_adj_omics1 = construct_fusion_graph(adj_spatial_omics1, adj_feature_omics1, alpha=alpha)
    fused_adj_omics2 = construct_fusion_graph(adj_spatial_omics2, adj_feature_omics2, alpha=alpha)



    adj = {
        'fused_adj_omics1': fused_adj_omics1,
        'fused_adj_omics2': fused_adj_omics2
           }
    # 返回的adj字典是四个个预处理后的 PyTorch 稀疏张量
    return adj


def transform_adjacent_matrix(adjacent):
    """邻接表 adj转换成邻接矩阵"""
    # 计算矩阵的维度
    n_spot = adjacent['x'].max() + 1  # n_spot=3484
    # 构建coo_matrix稀疏矩阵：保留了非零元素位置和数值
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    # coo_matrix稀疏矩阵形式的邻接矩阵
    return adj

def fix_seed(seed=2024):
    import os, random
    import torch, numpy as np
    import torch.backends.cudnn as cudnn

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)



def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    # X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    # adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    # print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    # 创建邻接矩阵
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)  # 将value的值全部设为1
    return adj