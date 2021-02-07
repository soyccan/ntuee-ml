from steel_defect_detector.utils import (
    get_label, get_img, get_data_path, add_weight_decay, bce_dice_loss,
    mask2rle, dice_channel_torch, dice_loss, dice_single_channel, rle2mask,
    run_length_decode, run_length_encode, weighted_bceloss
)
from steel_defect_detector.dataset import SteelDataset
from steel_defect_detector.train import Train, Train_plus
from steel_defect_detector.test import Test
from steel_defect_detector.validate import valid_test
from steel_defect_detector.model import Unet
import os
import logging


MODEL_PATH = '../working/best.pt'
SUBMISSION_PATH = '../working/submission.csv'

os.environ['PYTHONPATH'] = (
    os.environ['PYTHONPATH'] + ':' + os.path.dirname(__file__)
    if os.environ.get('PYTHONPATH')
    else os.path.dirname(__file__))

logging.basicConfig(level='DEBUG')
