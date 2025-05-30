import numpy as np
import pandas as pd
import sys
import os
from copy import deepcopy
from tqdm import tqdm
sys.path.append('./utils/')

from DD_data_extractor_git import Data_extractor_v5, normalize, bucketize_new, split_dataset2, flatten_2D_list, output_vars_v5, split_dataset_OddEven

#parameters --------------------------------------------------------
Data_saveName = 'multitrain_May28'
anatuple_path = "/home/debryas/data/HNL/anatuple/"
period = '2016_HIPM'
tag = 'AddJETcorr'
channels = ['tee', 'tem', 'tmm', 'tte', 'ttm']
singletrain = False
#-------------------------------------------------------------------
channel_dict = {'tee': 0, 'tem': 1, 'tmm': 2, 'tte': 3, 'ttm': 4}

features=[]
features.extend(deepcopy(output_vars_v5))
features.extend(['signal_label', 'channel', 'event_type', 'mass_hyp'])
flat_features = flatten_2D_list(features)
print(f'Number of inputs: {len(flat_features)}')

values = []
for i in range(len(flat_features)):
    values.append([])
data = dict(zip(flat_features, values))

for channel in tqdm(channels, desc='channels'):
    #print(f"Processing {channel}...")
    extractor = Data_extractor_v5(channel)
    data_currchannel = extractor(os.path.join(anatuple_path,period, tag) + '/' + channel + "/anatuple/", data=data)
    for key in data.keys():
        data[key].extend(data_currchannel[key])

data = pd.DataFrame(data)
weightNorm = deepcopy(data['genWeight'])
data['weightNorm'] = weightNorm

N = len(data['event'])
data_norm = normalize(pd.DataFrame(data), 'mass_hyp', N, weight_name='weightNorm')
data_norm = normalize(data_norm, 'signal_label', N, weight_name='weightNorm')
data_norm = normalize(data_norm, 'channel', N/5, weight_name='weightNorm')
data_processed, channel_indices = bucketize_new(data_norm, 'channel', class_names_dict=channel_dict)

output_dir = os.path.join(os.getcwd(),"saved_files", "extracted_data")
os.makedirs(output_dir, exist_ok=True)
data_processed.to_pickle(output_dir + f"/Data_{tag}_{period}_"+ Data_saveName)

if singletrain:
    train, val, test = split_dataset2(data_processed)
    pd.to_pickle(train, output_dir + f"/train_{tag}_{period}_" + Data_saveName)
    pd.to_pickle(val  , output_dir + f"/val_{tag}_{period}_"   + Data_saveName)
    pd.to_pickle(test , output_dir + f"/test_{tag}_{period}_"  + Data_saveName)
else:
    train1, train2, val1,val2 = split_dataset_OddEven(data_processed)
    pd.to_pickle(train1, output_dir + f"/train1_{tag}_{period}_" + Data_saveName)
    pd.to_pickle(train2, output_dir + f"/train2_{tag}_{period}_" + Data_saveName)
    pd.to_pickle(val1  , output_dir + f"/val1_{tag}_{period}_"   + Data_saveName)
    pd.to_pickle(val2  , output_dir + f"/val2_{tag}_{period}_"   + Data_saveName)
