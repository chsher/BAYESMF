import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


HOME_PATH = '/home/sxchao/crc_atlas/c222/colon10x_c222_subClean/'
IN_PATH = os.path.join(HOME_PATH, 'colon10x_c222_rawCount_042320.h5ad')


def make_insilico_dataset(alpha=2, D=1000, K=15, V=100, random_state=22690, return_assignments=False):
    if type(random_state) is int:
        np.random.seed(random_state)

    table_assignments = [0]
    seating_chart = [[0]]
    num_tables = 1
    for d in range(1, D):
        probs = [len(x) for x in seating_chart] + [alpha]
        probs /= sum(np.array(probs))

        table_assignment = np.random.choice(np.arange(num_tables + 1), p=probs)
        table_assignments.append(table_assignment)
        
        if table_assignment == num_tables:
            seating_chart.append([d])
            num_tables += 1
        else:
            seating_chart[table_assignment].append(d)
            
    # sample the topic-specific feature weights
    feature_weights = np.random.multivariate_normal(np.zeros(V) - 0.5, np.eye(V), K).T
    feature_weights[feature_weights < 0.0] = 0.0

    # sample the table-specific topic weights
    topic_weights = np.random.multivariate_normal(np.zeros(K) - 0.5, np.eye(K), num_tables).T
    topic_weights[topic_weights < 0.0] = 0.0
    
    # sample the occupant-specific gene expression
    sigma = np.eye(K)

    lambd = []
    for c,occupants in enumerate(seating_chart):
        mu = topic_weights[:, c]
        weights = np.random.multivariate_normal(mu, sigma, len(occupants)).T
        weights[weights < 0.0] = 0.0    
        lambd.append(np.matmul(feature_weights, weights))

    lambd = np.concatenate(lambd, axis=1)
    
    X = np.random.poisson(lambd)
    
    sparsity = np.count_nonzero(X == 0) / X.size
    
    if return_assignments:
        return table_assignments
    else:
        return X, sparsity


def make_downsampled_dataset(D=1000, V=100, random_state=22690, return_genes=False):
    if type(random_state) is int:
        np.random.seed(random_state)
    
    adata = sc.read_h5ad(IN_PATH)
    adata.var_names_make_unique()

    clusters = adata.obs['clFullc222'].unique()
    clusters_t = [x for x in clusters if x[0]=='T']
    adata_t = adata[adata.obs['clFullc222'].isin(clusters_t),:]

    # expression matrix for CD8+ T cells
    cd8 = np.asarray(adata_t[:,['CD8A','CD8B']].X.todense())
    cd8 = cd8.sum(axis=1) > 0
    exp = np.asarray(adata_t[cd8, :].X.todense())
    
    cells = np.count_nonzero(exp, axis=1)
    genes = np.count_nonzero(exp, axis=0)
    
    cells_want = np.random.choice(np.nonzero(cells > 2000)[0], D)
    genes_want = np.random.choice(np.nonzero(genes > 2000)[0], V)

    # downsampled expression matrix
    dexp = exp[cells_want, :]
    dexp = dexp[:, genes_want]
    X = dexp.T
    
    sparsity = np.count_nonzero(X == 0) / X.size
    
    if return_genes:
        return adata_t.var.index[genes_want]
    else:
        return X, sparsity
