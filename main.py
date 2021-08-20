import os
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn 


from IPython.display import display
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from scipy.stats import skew, norm, probplot
import seaborn as sns
from item_similarity_recommender import ItemSimilarityRecommender 
from popularity_recommender import PopularityRecommender



# sns.set(style="ticks", color_codes=True, font_scale=1.5)
# color = sns.color_palette()
# sns.set_style('darkgrid')

track_metadata_df = pd.read_csv('./song_data.csv')
count_play_df = pd.read_csv('./10000.txt', sep='\t', header=None, names=['user','song','play_count'])
user_song_list_count=pd.read_csv('./data/preprocessed.csv')



################# POPULARITY RECOMMENDER########################
def create_popularity_recommendation(train_data, user_id, item_id, n=10):
    # Recommendation score: count of user_ids for each unique song
    train_data_grouped = train_data.groupby([item_id]).agg({user_id: 'count'}).reset_index()
    train_data_grouped.rename(columns = {user_id: 'score'},inplace=True)
    
    #Sort based on recommendation score
    train_data_sort = train_data_grouped.sort_values(['score', item_id], ascending = [0,1])
    
    #Recommendation rank
    train_data_sort['Rank'] = train_data_sort.score.rank(ascending=0, method='first')
        
    # top n recommendations
    popularity_recommendations = train_data_sort.head(n)
    return popularity_recommendations

def popularity_recommender(n):
    recommendations = create_popularity_recommendation(user_song_list_count,'user','title', n)
    display(recommendations)

# popularity_recommender(10)
##################################################################################


####################### Item Similarity Based RECOMMENDER#########################
def total_listen_5000_songs():
    total_play_count = sum(user_song_list_count.listen_count)
    play_count = user_song_list_count[['song', 'listen_count']].groupby('song').sum().\
                sort_values(by='listen_count',ascending=False).head(5000)

    print('5,000 most popular songs represents {:3.2%} of total listen.'.format(float(play_count.sum())/total_play_count))

    song_subset = list(play_count.index[:5000])
    user_subset = list(user_song_list_count.loc[user_song_list_count.song.isin(song_subset), 'user'].unique())
    user_song_list_count_sub = user_song_list_count[user_song_list_count.song.isin(song_subset)]
    display(user_song_list_count_sub.head())

# total_listen_5000_songs()

def item_similarity_based_recommender(n):
    total_play_count = sum(user_song_list_count.listen_count)
    play_count = user_song_list_count[['song', 'listen_count']].groupby('song').sum().\
                sort_values(by='listen_count',ascending=False).head(5000)
    song_subset = list(play_count.index[:5000])
    user_subset = list(user_song_list_count.loc[user_song_list_count.song.isin(song_subset), 'user'].unique())
    user_song_list_count_sub = user_song_list_count[user_song_list_count.song.isin(song_subset)]
    is_model = ItemSimilarityRecommender()
    is_model.create(user_song_list_count_sub, 'user', 'title')
    user_id = list(user_song_list_count_sub.user)[n]
    user_items = is_model.get_user_items(user_id)

    #Recommend songs for the user using personalized model
    is_model.recommend(user_id)

# item_similarity_based_recommender(7)

##################################################################################

####################### Matrix Factorization Based RECOMMENDER####################

def compute_svd(urm, K): # Uses svds function to break down the utility matrix into three different matrices
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt

def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test): # Provide Predictions  
    rightTerm = S*Vt 
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID,max_recommendation ), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings

def show_recomendations(uTest, num_recomendations = 10):
    for user in uTest:
        print('-'*70)
        print("Recommendation for user id {}".format(user))
        rank_value = 1
        i = 0
        while (rank_value <  num_recomendations + 1):
            so = uTest_recommended_items[user,i:i+1][0]
            if (small_set.user[(small_set.so_index_value == so) & (small_set.us_index_value == user)].count()==0):
                song_details = small_set[(small_set.so_index_value == so)].\
                    drop_duplicates('so_index_value')[['title','artist_name']]
                print("The number {} recommended song is {} BY {}".format(rank_value, 
                                                                    list(song_details['title'])[0],
                                                                    list(song_details['artist_name'])[0]))
                rank_value+=1
            i += 1

def matrix_factorization_recommender(uTest):

    # Adding rating to the data by replacing the play count with a fractional play count
    # The "likeness" for a song is in the range of [0,1]
    user_song_list_listen = user_song_list_count[['user','listen_count']].groupby('user').sum().reset_index()
    user_song_list_listen.rename(columns={'listen_count':'total_listen_count'},inplace=True)
    user_song_list_count_merged = pd.merge(user_song_list_count,user_song_list_listen)
    user_song_list_count_merged['fractional_play_count'] = \
    user_song_list_count_merged['listen_count']/user_song_list_count_merged['total_listen_count']


    #Convert the dataframe into a numpy matrix in the format of utility matrix
    user_codes = user_song_list_count_merged.user.drop_duplicates().reset_index()
    user_codes.rename(columns={'index':'user_index'}, inplace=True)
    user_codes['us_index_value'] = list(user_codes.index)

    song_codes = user_song_list_count_merged.song.drop_duplicates().reset_index()
    song_codes.rename(columns={'index':'song_index'}, inplace=True)
    song_codes['so_index_value'] = list(song_codes.index)

    small_set = pd.merge(user_song_list_count_merged,song_codes,how='left')
    small_set = pd.merge(small_set,user_codes,how='left')
    mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]

    data_array = mat_candidate.fractional_play_count.values
    row_array = mat_candidate.us_index_value.values
    col_array = mat_candidate.so_index_value.values

    data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)

    display(data_sparse)

    K=50
    urm = data_sparse
    MAX_PID = urm.shape[1]
    MAX_UID = urm.shape[0]

    U, S, Vt = compute_svd(urm, K)
    uTest = [4,5,6,7,8,873,23]

    uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)

    show_recomendations(uTest)

matrix_factorization_recommender([4,5,6,7,8,873,23])


##################################################################################

