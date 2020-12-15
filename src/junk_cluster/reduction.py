"""# Dimension Reduction & Clustering"""
import numpy as np


def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    # 因為是 binary unsupervised clustering，因此取 max(acc, 1-acc)
    return max(acc, 1-acc)


from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding


def reduce_dim_pca(latents, **kwargs):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=4)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # Second Dimension Reduction
    X_embedded = TSNE(n_components=2, **kwargs).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    return X_embedded

def reduce_dim_lle(latents, **kwargs):
    # First Dimension Reduction
    transformer = LocallyLinearEmbedding(n_components=200, n_jobs=4)
    lap = transformer.fit_transform(latents)
    print('First Reduction Shape:', lap.shape)

    # Second Dimension Reduction
    X_embedded = TSNE(n_components=2, **kwargs).fit_transform(lap)
    print('Second Reduction Shape:', X_embedded.shape)

    return X_embedded


def reduce_dim_laplacian(latents, **kwargs):
    # First Dimension Reduction
    transformer = SpectralEmbedding(n_components=200, affinity='rbf', n_jobs=4)
    lap = transformer.fit_transform(latents)
    print('First Reduction Shape:', lap.shape)

    # Second Dimension Reduction
    X_embedded = TSNE(n_components=2, **kwargs).fit_transform(lap)
    print('Second Reduction Shape:', X_embedded.shape)

    return X_embedded


import torch
from torch.utils.data import DataLoader
from junk_cluster.preprocess import *
from junk_cluster.dataset import *


def inference(X, model, batch_size=256):
    """ Get latent code from auto-encoder """
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x)
        vec = vec.view(img.size()[0], -1).detach().numpy()
        if i == 0:
            latents = vec
        else:
            latents = np.concatenate((latents, vec), axis=0)
    print('Latents Shape:', latents.shape)
    return latents
