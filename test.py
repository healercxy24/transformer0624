# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process import *
from model import *
import optuna


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')
torch.manual_seed(1)
torch.cuda.manual_seed(2)

model = torch.load('F001_DAST_prediciton_model_11pk1')
print(model)