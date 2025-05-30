import os
import pandas as pd
from copy import deepcopy
import sys

sys.path.append('./utils/')
sys.path.append('./DNN/')
sys.path.append('./SignificancePlotting/')
from DD_data_extractor_git import output_vars_v4, flatten_2D_list
from Significance_func import process_dataframe, plot_average_significance_withpd_evenOdd

#parameters --------------------------------------------------------
anatuple_path = "/home/debryas/data/HNL/anatuple/"
periods = ['2018','2017','2016', '2016_HIPM']
tag = 'AddJETcorr'
Data_saveName = 'multitrain_May28'
channels = ['tte', 'ttm'] # , 'tte', 'ttm' 'tee', 'tem', 'tmm'
save_folder= 'simple_dnn_allperiods'
namefig = 'average_significance_on_ttl_channels.pdf'
#-------------------------------------------------------------------

features = deepcopy(output_vars_v4)
features.extend(['signal_label', 'channel', 'event_type', 'mass_hyp'])
channel_dict = {'tee': 0, 'tem': 1, 'tmm': 2, 'tte': 3, 'ttm': 4}
channels_nb = [channel_dict[ch] for ch in channels]
values = []
flat_features = flatten_2D_list(features)
for i in range(len(flat_features)):
    values.append([])
data = dict(zip(flat_features, values))

current_dir = os.getcwd()
output_dir = os.path.join(current_dir,"saved_files", "extracted_data")
DATA = pd.DataFrame()
for period in periods:
    dir = os.path.join(output_dir, f"Data_{tag}_{period}_{Data_saveName}")
    data = pd.read_pickle(dir)
    data = data[data["channel"].isin(channels_nb)]
    DATA = pd.concat([DATA, data], ignore_index=True)

flat_feat=DATA.columns.to_list()
flat_feat.remove('signal_label')

# Call the function
def get_channel_map(channels, channel_dict):
    return {ch: channel_dict[ch] for ch in channels if ch in channel_dict}
channels_data = process_dataframe(DATA, flat_feat, get_channel_map(channels, channel_dict))

All_channel_dict = channels_data

model_info_list = {}
parity = ['even', 'odd']
for par in parity:
    model_info_dir=os.path.join(current_dir, 'saved_files', 'saved_models', f'{save_folder}_{par}')
    model_info_list[par] =pd.read_pickle(os.path.join(model_info_dir, 'model_info_list'))

output_path_fig = os.path.join(current_dir, 'SignificancePlotting', 'figures', namefig) 
plot_average_significance_withpd_evenOdd(All_channel_dict, ['Mt_tot', 'scores'], model_info_list, 'binmaker_rightleft', output_path_fig, X=0.15, hide_errorbars=False)