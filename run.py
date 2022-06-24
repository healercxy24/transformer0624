# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:34:13 2022

@author: njucx
"""

import torch
import torch.nn as nn
from data_process import *
import optuna


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
study = optuna.create_study(study_name='test_loss_optim', direction="minimize", storage = "sqlite:///20220605_3.db", load_if_exists=True)  # Create a new study.
df = study.best_trial.params
learning_rate = df['learning_rate']
seq_len = df['seq_len']
nlayers = df['nlayers']
dropout = df['dropout']
nhid = df['nhid']
seq_len = df['seq_len']
'''

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss
    
    
batch_size = 32

dataset_name = 'FD001'
dataset = get_dataset(dataset_name, 50);
test_seq = dataset['lower_test_seq_tensor']
test_label = dataset['lower_test_label_tensor']


model = torch.load('temp_model_FD001_18.pk1').to(device)
criterion = RMSELoss()


def test(model, criterion):

    model.eval()  # turn on evaluation mode

    total_test_loss = 0
    pre_result = []  # list(101) -> (50, 128, 1)
    num_batches = test_seq.shape[0] // batch_size
    src_mask = generate_square_subsequent_mask(18).to(device)
      
    
    with torch.no_grad():

        for batch, i in enumerate(range(0, num_batches*batch_size, batch_size)):
            # compute the loss for the lower-level
            inputs, targets = get_batch(test_seq, test_label, i, batch_size)
            inputs = inputs.permute(2, 1, 0)  # [18, 32, 50]
            targets = targets.reshape(1, batch_size, 50)  #[1, 32, 50]
            predictions = model(inputs, targets, src_mask)
            loss = criterion(predictions, targets)               
            
            total_test_loss += loss.item()
            pre_result.append(np.array(predictions.cpu()))
            
        total_test_loss = total_test_loss / num_batches
        

    return total_test_loss, pre_result


test_loss , _ = test(model, criterion)


print("best test loss:", test_loss)