# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 18:37:08 2022

@author: cxy
"""

import pandas as pd
import torch
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')

#%% add the RUL column

def add_RUL(dataset):
    rul = pd.DataFrame(dataset.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    dataset = dataset.merge(rul, on=['id'], how='left')
    # add a column 'RUL'
    rul = dataset['max'] - dataset['cycle']
    # drop 'max' column
    dataset.drop('max', axis=1, inplace=True)
    

    for index, row in rul.iteritems():
        if(row > 130):
            rul[index] = 130

    
    dataset['RUL'] = rul
    
    return dataset


def add_RUL_test(dataset, truth_rul):
    rul = pd.DataFrame(dataset.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    rul['max'] = rul['max'] + truth_rul['RUL']
    dataset = dataset.merge(rul, on=['id'], how='left')
    # add a column 'RUL'
    rul = dataset['max'] - dataset['cycle']
    # drop 'max' column
    dataset.drop('max', axis=1, inplace=True)
    

    for index, row in rul.iteritems():
        if(row > 130):
            rul[index] = 130

    
    dataset['RUL'] = rul
    
    return dataset
    
#%% training set processing

def dataset_process(dataset_name):
    """
    Parameters:
        dataset_name : string : 'FD001', 'FD002', 'FD003', 'FD004'
    return:
        train_set : [100,] --> [20631, 26]
        test_set : [100,] --> [20631, 26]
        in total 26 features(including : id, cycle, 3 setting operations, 21 sensor datas)
    """
    
    root_path = './CMAPSSDataNASA/'
    
    # set the column names
    title_names = ['id', 'cycle']
    setting_names = ['setting1', 'setting2', 'setting3']
    data_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = title_names + setting_names + data_names
    
    drop_cols = ['setting3', "s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
    
    # load data from the txt file
    train_df = pd.read_csv((root_path + 'train_' + dataset_name + '.txt'), sep='\s+', header=None, names=col_names)
    # First sort by the elements in the 'id' column, and when the elements in the 'id' column are the same, sort by the 'cycle' column
    train_df = train_df.sort_values(['id', 'cycle'])
    
    test_df = pd.read_csv((root_path + 'test_' + dataset_name + '.txt'), sep='\s+', header=None, names=col_names)
    test_df = test_df.sort_values(['id', 'cycle'])
    
    rul_df = pd.read_csv((root_path + 'RUL_' + dataset_name + '.txt'), sep='\s+', header=None, names=['RUL'])
    rul_df['id'] = rul_df.index + 1
    
    train_df.drop(drop_cols, axis=1, inplace=True)
    test_df.drop(drop_cols, axis=1, inplace=True)
    
    
    '''process the train data'''
    title = train_df.iloc[:, 0:2]
    data = train_df.iloc[:, 2:]
    
    # minmaxscaler for data
    data_norm = (data - data.min())/(data.max() - data.min()) 
    data_norm = data_norm.fillna(0) # replace all the NaN with 0    
    # add RUL col to title
    title = add_RUL(title)    
    # merge title & data_norm
    train_data = pd.concat([title, data_norm], axis=1)   # [20631, 27]     
    # group the training set with 'id'
    #train_group = train_data.groupby(by="id")
    
    
    '''process the test data'''
    title = test_df.iloc[:, 0:2]
    data = test_df.iloc[:, 2:]
    
    # minmaxscaler for data
    data_norm = (data - data.min())/(data.max() - data.min()) 
    data_norm = data_norm.fillna(0) # replace all the NaN with 0
    # add RUL col to title
    title = add_RUL_test(title, rul_df)
    
    # merge title & data_norm
    test_data = pd.concat([title, data_norm], axis=1)   # [13096, 27]     
    # group the training set with 'id'
    #test_group = test_data.groupby(by="id")
    
    return train_data, test_data, rul_df

#%% PCA processing with numpy
import numpy as np


def pca(input_data, M): # M is the components you want
    """
    input:
        X: ->narray of float64;
        M: ->int: the number of PCA features
    return:
        data: ->narrya of float64;
    """
    
    input_data = input_data.values
    #mean of each feature
    n_samples, n_features = input_data.shape
    mean=np.array([np.mean(input_data[:,i]) for i in range(n_features)])
    #normalization
    norm_input = input_data-mean
    #scatter matrix
    scatter_matrix=np.dot(np.transpose(norm_input),norm_input)
    #Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    [x[0] for x in eig_pairs].sort(reverse=True)
    # select the top k eig_vec
    feature=np.array([ele[1] for ele in eig_pairs[:M]])
    #get new data
    data=np.dot(norm_input,np.transpose(feature))
    # data=pd.DataFrame(data)
    # data.columns=['s1','s2','s3','s4']
    
    return data


#%% slicing window to get sequences for lower level

# generate the sequence according to the time step
def gen_sequence(data, seq_len):
    
    num_elements = data.shape[0]
    data = data.to_numpy()
    for start, stop in zip(range(0, num_elements - seq_len), range(seq_len, num_elements)):
        yield data[start:stop, :]

def gen_labels(data, seq_len):
    
    num_elements = data.shape[0]
    for start, stop in zip(range(0, num_elements - seq_len), range(seq_len, num_elements)):
        yield data[start:stop]


#%% get processed dataset

"""
loss the number of seq_len data because of gen_sequence : 
    trainset : 20631 = 16000 + 4581 + seq_len

training set:
    train_seq_tensor : torch.Size([16000, 26])
    train_label_tensor : torch.Size([16000])
    valid_seq_tensor : torch.Size([4631, 26])
    valid_label_tensor : torch.Size([4631])


testing set: 
    test_seq_tensor : torch.Size([13096, 26])
    test_label_tensor : torch.Size([13096])
"""


def get_dataset(dataset_name, seq_len):   
    
    train_data, test_data, _ = dataset_process(dataset_name)  
    
    '''generate sequences for train data'''   
    # generate labels for train
    train_label = train_data['RUL'].to_numpy()
    train_label = list(gen_labels(train_label, seq_len))
    label_tensor = torch.tensor(train_label)   # torch.Size([20581, 50])
    label_tensor = label_tensor.float().to(device)
    
    train_data.drop(train_data.columns[2], axis=1, inplace=True) # can't put RUL as a feature for training
    #train_data = pca(train_data, 8)     
    seq_array = list(gen_sequence(train_data, seq_len))  
    seq_tensor = torch.tensor(seq_array)    # [20581, 50, 18]
    seq_tensor = seq_tensor.float().to(device)   

    """
    # split into train&valid
    seq_list = np.array(seq_tensor.cpu()) # [20581, 50, 26]
    label_list = np.array(label_tensor.cpu()) # [20581, 50]
    
    np.random.seed(123)
    temp = np.arange(0, len(seq_list)) # [20581,]
    np.random.shuffle(temp)
    
    # shuffle the index
    new_seq_tensor = []
    new_label_tensor = []
    for i in temp:
        new_seq_tensor.append(seq_list[i])
        new_label_tensor.append(label_list[i])
    
    # get new tensor according to new index
    new_seq_tensor = np.array(new_seq_tensor)
    new_label_tensor = np.array(new_label_tensor)
    new_seq_tensor = torch.tensor(new_seq_tensor)
    new_label_tensor = torch.tensor(new_label_tensor)
    """

    
    #split the new dataset
    train_seq_tensor = seq_tensor[0:16000,:].to(device)
    train_label_tensor = label_tensor[0:16000].to(device)
    valid_seq_tensor = seq_tensor[16000:,: ].to(device)
    valid_label_tensor = label_tensor[16000:].to(device)
    
    
    '''process data for test dataset'''
    # generate labels
    test_label = test_data['RUL'].to_numpy()
    test_label = list(gen_labels(test_label, seq_len))   # [13046, 50]
    test_label_tensor = torch.tensor(test_label)
    test_label_tensor = test_label_tensor.float().to(device)
    
    test_data.drop(test_data.columns[2], axis=1, inplace=True) # can't put RUL as a feature for training
    #test_data = pca(test_data, 8)
    test_array = list(gen_sequence(test_data, seq_len))   # [13046, 50, 26]
    test_seq_tensor = torch.tensor(test_array)
    test_seq_tensor = test_seq_tensor.float().to(device) 
    
    
    dataset = {'lower_train_seq_tensor' : train_seq_tensor,
               'lower_train_label_tensor' : train_label_tensor,
               #'upper_train_seq_tensor' : train_seq_tensor2,
               #'upper_train_label_tensor' : train_label_tensor2,
               'lower_valid_seq_tensor' : valid_seq_tensor,
               'lower_valid_label_tensor' : valid_label_tensor,
               #'upper_valid_seq_tensor' : valid_seq_tensor2,
               #'upper_valid_label_tensor' : valid_label_tensor2,
               'lower_test_seq_tensor' : test_seq_tensor,
               'lower_test_label_tensor' : test_label_tensor,
               #'upper_test_seq_tensor' : upper_test_seq_tensor,
               #'upper_test_label_tensor' : upper_test_label_tensor
               }
    
    return dataset

#%% get batch

def get_batch(data_source, truth_source, i, batch_size):

    """

    Args:

        source: Tensor, shape [dataset_length, sequence_length, num_features]

        i: int
        
        seq_len : size of sequence
        
        d_model : the embed dim (required) --> the number of input_features

 

    Returns:

        tuple (data, target), where data has shape [sequence_length, batch_size, num_features] and

        target has shape [seq_len, batch_size]

    """

    data = data_source[i:i+batch_size, :]
    data = data.view(-1, batch_size, data.shape[2]).contiguous()

    target = truth_source[i:i+batch_size].permute(1, 0)

    return data, target


#%% generate mask for input

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.
    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)