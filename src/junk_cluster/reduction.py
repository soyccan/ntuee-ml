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

def reduce_dim(latents):
    # First dimension reduction
    # emb1 = KernelPCA(n_components=512, n_jobs=16, kernel='rbf').fit_transform(latents)
    # np.save('emb1.npy', emb1)
    # emb1 = np.load('emb1.npy')
    emb1 = latents
    # print('First Reduction Shape:', emb1.shape)

    # Second dimension reduction
    # TODO: TNSE do not support n_jobs, remove it on production
    # X_embedded = TSNE(n_components=2,
    #                   perplexity=50,
    #                   n_jobs=16,
    #                   verbose=2).fit_transform(emb1)
    # X_embedded = LocallyLinearEmbedding(n_components=2,
    #                                     n_jobs=16).fit_transform(latents)
    X_embedded = SpectralEmbedding(n_components=2,
                                   affinity='rbf',
                                   n_jobs=16).fit_transform(latents)
    print('Second Reduction Shape:', X_embedded.shape)
    np.save('emb.npy', X_embedded)

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
        x = torch.FloatTensor(x).cuda()
        vec, img = model(x)
        vec = vec.view(img.size()[0], -1).cpu().detach().numpy()
        if i == 0:
            latents = vec
        else:
            latents = np.concatenate((latents, vec), axis=0)
    print('Latents Shape:', latents.shape)
    return latents
