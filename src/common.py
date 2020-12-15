""" Common definitions """

MODEL_PATH = './checkpoints/best.pth'
# MODEL_PATH = '../model/0.82888.pth'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + os.path.dirname(__file__) \
                           if os.environ.get('PYTHONPATH') \
                           else os.path.dirname(__file__)

from junk_cluster.dataset import *
from junk_cluster.model import *
from junk_cluster.preprocess import *
from junk_cluster.reduction import *
from junk_cluster.util import *
