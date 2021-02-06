"""# Prepare Training Data

定義我們的 preprocess：將圖片的數值介於 0~255 的 int 線性轉為 -1～1 的 float。
"""
import numpy as np


def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list


def postprocess(image_list):
    """ Inverse of preprocess() """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 2, 3, 1))
    image_list = (image_list + 1) / 2 * 255.0
    image_list = image_list.astype(np.uint8)
    return image_list
