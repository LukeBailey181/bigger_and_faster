import os
import shutil

from argparse import ArgumentParser
from pathlib import Path

import json
import random
import numpy as np
from collections import namedtuple
import time
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import pickle
import collections

# This is used for running on Huawei Cloud.
oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_super_kd import SuperTinyBertForPreTraining, SuperBertForPreTraining, BertConfig
from transformer.modeling_base import BertModel
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from utils import sample_arch_4_kd, sample_arch_4_mlm

from quantize_utils import quantize_model

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import logging


def main():
    parser = ArgumentParser()
    parser.add_argument('--super_model', type=str, default='8layer_bert', required=True, help="Directory to the saved super model")
    args = parser.parse_args()

    config = BertConfig.from_pretrained(os.path.join(args.super_model, CONFIG_NAME))
    super_model = SuperTinyBertForPreTraining.from_pretrained(args.super_model, config)
    super_model.to('cpu')
    print("Pretrained Super Model Loaded!")

    # Quantizing super model
    super_model_quant = quantize_model(super_model)
    print("Pretrained Super Model Quantized!")
    
    saving_path_quant = args.super_model + '_INT8'
    if not os.path.isdir(saving_path_quant):
        os.mkdir(saving_path_quant)

    shutil.copytree(args.super_model, saving_path_quant)
    output_model_file = os.path.join(saving_path_quant, WEIGHTS_NAME)
    torch.save(super_model_quant.state_dict(), output_model_file)


if __name__ == '__main__':
    main()