import os
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn


from IPython.display import display
import pandas as pd
# import numpy as np
# import time
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from scipy.sparse import coo_matrix
# import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
# from scipy.sparse.linalg import svds
# from scipy.sparse import csc_matrix
# from scipy.stats import skew, norm, probplot
# import seaborn as sns

sns.set(style="ticks", color_codes=True, font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')

track_metadata_df = pd.read_csv('./data/song_data.csv')
count_play_df = pd.read_csv('./data/10000.txt', sep='\t', header=None, names=['user','song','play_count'])

# print('First see of track metadata:')
# print('Number of rows:', track_metadata_df.shape[0])
# print('Number of unique songs:', len(track_metadata_df.song_id.unique()))
# display(track_metadata_df.head())
# print('Note the problem with repeated track metadata. Let\'s see of counts play song by users:')
# display(count_play_df.shape, count_play_df.head())

unique_track_metadata_df = track_metadata_df.groupby('song_id').max().reset_index()

# print('Number of rows after unique song Id treatment:', unique_track_metadata_df.shape[0])
# print('Number of unique songs:', len(unique_track_metadata_df.song_id.unique()))
# display(unique_track_metadata_df.head())

user_song_list_count = pd.merge(count_play_df, 
                                unique_track_metadata_df, how='left', 
                                left_on='song', 
                                right_on='song_id')
user_song_list_count.rename(columns={'play_count':'listen_count'},inplace=True)
del(user_song_list_count['song_id'])

user_song_list_count.to_csv('./data/preprocessed.csv')
