import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import os
import logging
# import pyvips


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# %%

sigmoid = nn.Sigmoid()


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': l2_value}]


def dice_channel_torch(probability, truth, threshold):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(
                    probability[i, j, :, :],
                    truth[i, j, :, :],
                    threshold[j])
                mean_dice_channel += channel_dice / (batch_size * channel_num)
    return mean_dice_channel


def dice_single_channel(probability, truth, threshold, eps=1E-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return dice


def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height, width), np.float32)
    if rle != '':
        mask = mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start, length in r:
            start = start - 1  # ???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask = mask.reshape(width, height).T
    return mask


def run_length_encode(mask):
    # possible bug for here
    m = mask.T.flatten()
    if m.sum() == 0:
        rle = ''
    else:
        m = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = ' '.join(str(r) for r in run)
    return rle


def dice_loss(y_pred, y_true):
    weight = [1, 1, 1, 1]
    smooth = 1e-9
    y_pred = sigmoid(y_pred)
    y_true_f = y_true.view(-1, 4, 256, 1600)
    y_pred_f = y_pred.view(-1, 4, 256, 1600)
    score = 0
    batch_size = y_true_f.shape[0]
    channel_num = y_true_f.shape[1]
    for i in range(batch_size):
        for j in range(channel_num):
            intersection = y_true_f[i, j, :, :] * y_pred_f[i, j, :, :]
            score += weight[j] * ((2. * intersection.sum() + smooth) / (
                y_true_f[i, j, :, :].sum() + y_pred_f[i, j, :, :].sum() + smooth))
    score /= batch_size
    return - score


def bce_dice_loss(y_pred, y_true):
    return weighted_bceloss(y_pred, y_true) + dice_loss(y_pred, y_true)


def weighted_bceloss(y_pred, y_true):
    weight = [1, 1, 1, 1]
    loss = 0
    bceloss = nn.BCEWithLogitsLoss()
    for c in range(4):
        loss += bceloss(y_pred[:, c, :, :], y_true[:, c, :, :]) * weight[c]
    return loss


def get_data_path(folder):
    X_path = []
    # print(os.listdir(folder))
    for filename in os.listdir(folder):
        # print(filename)
        X_path.append(filename)

    X_path = np.array(X_path)
    return X_path


def get_label(path):
    df = pd.read_csv(path)
    df['EncodedPixels1'] = df[df['ClassId'] == 1]['EncodedPixels']
    df['EncodedPixels2'] = df[df['ClassId'] == 2]['EncodedPixels']
    df['EncodedPixels3'] = df[df['ClassId'] == 3]['EncodedPixels']
    df['EncodedPixels4'] = df[df['ClassId'] == 4]['EncodedPixels']
    df.drop(['ClassId', 'EncodedPixels'], axis=1, inplace=True)
    get_non_nan = lambda s: [x for x in s if x][0]
    df = df.groupby('ImageId').agg({'EncodedPixels1': get_non_nan,
                                    'EncodedPixels2': get_non_nan,
                                    'EncodedPixels3': get_non_nan,
                                    'EncodedPixels4': get_non_nan})
    df.fillna('', inplace=True)
    y = df.to_dict('index')
    for k in y.keys():
        y[k] = [y[k]['EncodedPixels1'], y[k]['EncodedPixels2'],
                y[k]['EncodedPixels3'], y[k]['EncodedPixels3']]
    return y


def get_img(path):
    # with Image.open(path) as f:
    #     img = np.array(list(f.getdata()))
    # f.close()
    # # print('img shape',len(img),img[:,1])
    # return img[:, 1].reshape(256, 1600)

    # (H, W, C) -> (C, H, W)
    return np.array(Image.open(path))[:, :, 0].reshape((1, 256, 1600))
