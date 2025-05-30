import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from copy import deepcopy
import yaml
import os
import pickle
from pytorch2python import create_data_loaders, train_model #, test_model, create_test_loader
from DNN_models import DNN_bestFeature, DNN_flexible
#from torchsummary import summary

#import itertools
from tqdm import tqdm

"""
This module implements a Neural Architecture Search (NAS) algorithm for fully 
connected Deep Neural Networks (DNN).
It trains models using all input features defined in the 'selectionlonger' list.
The architecture of each model, defined by the depth and width of the network, 
is determined by the variable 'hidden_layer_configs'.

Main steps of the algorithm:

Load training, validation, and test datasets from specified pickle files.
The datasets are expected to be Pandas DataFrame objects.
Create PyTorch data loaders for these datasets.
These loaders are used to provide data to the model during training and testing phases.
Initialize a model with a specific configuration of hidden layers and assign it to the appropriate computing device (GPU if available, else CPU).
Define a Binary Cross-Entropy (BCE) loss function and an Adam optimizer.
Train the model for 1000 epochs, saving the model parameters that yield the best performance on the validation set.
Repeat the process for each configuration defined in 'hidden_layer_configs'.

At the end of the process, a DataFrame named 'model_info_list' is saved as a pickle file. 
This DataFrame contains the following information for each trained model: save_path (the path where the model is saved), 
save_name (the name of the saved model file), model_info (a string representation of the model's architecture), 
input_variables (the list of input variables used for the model), hidden_layers (the configuration of hidden layers in the model), 
and scaler_path (the path of the saved data scaler file used for feature normalization).
"""
channel_dict = {'tee': 0, 'tem': 1, 'tmm': 2, 'tte': 3, 'ttm': 4}

#current_dir = os.getcwd()
#parent_dir = os.path.dirname(current_dir)
parent_dir = os.getcwd()

with open(os.path.join(parent_dir, 'DNN',"dnnconfig.yml"), 'r') as stream:
    config = yaml.safe_load(stream)

hidden_layer_configs = config['hidden_layer_configs']
save_folder = config['save_folder']
save_name = config['save_name']
scaler_filename = config['scaler_filename']
selectionlonger = config['selectionlonger']
model_info_list_name = config['model_info_list_name']
#train_data_name = config['train_data_name']
#val_data_name = config['val_data_name']
#test_data_name = config['test_data_name']
saveName_data = config['saveName_data']
tag = config['tag']
channels = config['channels']
periods = config['periods']
channels = config['channels']
channels_nb = [channel_dict[ch] for ch in channels]
even_or_odd = config['even_or_odd']

# Data loading part
cdpath=os.path.join(parent_dir, 'saved_files', 'extracted_data')
print(cdpath)

if even_or_odd == 'even':
    nb = '1'
if even_or_odd == 'odd':
    nb = '2'

train = pd.DataFrame()
val = pd.DataFrame()
for period in periods:
    train_data_name= f'train{nb}_{tag}_{period}_{saveName_data}'
    val_data_name  = f'val{nb}_{tag}_{period}_{saveName_data}'
    train_period = pd.read_pickle(cdpath + '/'+ train_data_name)
    val_period = pd.read_pickle(cdpath + '/'+ val_data_name)
    train_period = train_period[train_period["channel"].isin(channels_nb)]
    val_period = val_period[val_period["channel"].isin(channels_nb)]
    train = pd.concat([train, train_period], ignore_index=True)
    val = pd.concat([val, val_period], ignore_index=True)

#test = pd.read_pickle(cdpath + '/'+ test_data_name)

train_loader, val_loader, scaler = create_data_loaders(train, val, selectionlonger)
#test_loader = create_test_loader(test, selectionlonger, scaler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features = deepcopy(selectionlonger)
features.remove('signal_label')
features.remove('weightNorm')

model = DNN_bestFeature(features).to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters())

save_path=os.path.join(parent_dir, 'saved_files', 'saved_models', save_folder)
os.makedirs(save_path, exist_ok=True)
save_name = '/'+save_name

model_info_list=pd.DataFrame()
input_vars = selectionlonger

pbar = tqdm(enumerate(hidden_layer_configs), desc='layer configs', total=len(hidden_layer_configs))
for i, hidden_layer_sizes in pbar:
    desc = 'layer config: ' + str(hidden_layer_sizes)
    
    train_loader, val_loader, scaler = create_data_loaders(train, val, input_vars)
    #test_loader = create_test_loader(test, input_vars, scaler)

    model = DNN_flexible(input_vars, hidden_layer_sizes).to(device)
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters())
    
    save_path=os.path.join(parent_dir, 'saved_files', 'saved_models', save_folder)
    scaler_save_path=os.path.join(save_path, 'scalers')
    os.makedirs(scaler_save_path, exist_ok=True)
    save_nameiter = save_name+str(i+1)

    scaler_filename_iter=scaler_filename+str(i+1)
    with open(os.path.join(scaler_save_path, scaler_filename_iter), 'wb') as f:
        pickle.dump(scaler, f)

    # Train the model
    model = train_model(train_loader, val_loader, model, optimizer, criterion, device, save_path, save_nameiter, epochs=1000)

    pbar.set_description(desc)
    # Append info to the list
    new_row = pd.DataFrame({
    'save_path': [save_path], 
    'save_name': [save_nameiter], 
    'model_info': [str(model)], 
    'input_variables': [input_vars], 
    'hidden_layers': [hidden_layer_sizes],
    'scaler_path': [os.path.join(scaler_save_path, scaler_filename_iter)]
    })
    
    model_info_list = pd.concat([model_info_list, new_row], ignore_index=True)

model_info_list.to_pickle(save_path + '/' + model_info_list_name)
