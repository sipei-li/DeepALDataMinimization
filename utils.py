import numpy as np
import pandas as pd
import scipy.sparse as sp

def get_top_k_scored_items(scores, top_k, sort_top_k=False):
    """Extract top K items from a matrix of scores for each user-item pair, optionally sort results per user.
    
    Args:
        scores: numpy.ndarray, score matrix of shape (n_users, n_items)
        top_k: int, number of top items to recommend
        sort_top_k: bool, flag to sort top k results
    
    Returns:
        numpy.ndarray, numpy.ndarray:
        - indices into score matrix for each user's top items
        - scores corresponding to top items
    """

    if isinstance(scores, sp.spmatrix):
        scores = scores.todense()
    
    if scores.shape[1] < top_k:
        raise Warning("Number of items is less than `top_k`, limiting `top_k` to number of items.")

    k = min(top_k, scores.shape[1])

    test_user_idx = np.arange(scores.shape[0])[:, None]

    # get top K items and scores
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = scores[test_user_idx, top_items]

    if sort_top_k:
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]
    
    return np.array(top_items), np.array(top_scores)


def nCoreReduction(df, n_core=5):
    n_core = 5

    init_df = df 
    init_shp = df.shape[0]
    filt_shp = None 

    while True:

        item_value_counts = init_df['item_id'].value_counts()
        filter_items = item_value_counts[item_value_counts > n_core].index.to_list()

        user_value_counts = init_df['user_id'].value_counts() 
        filter_users = user_value_counts[user_value_counts > n_core].index.to_list() 

        mask_filter_items = init_df['item_id'].isin(filter_items)
        mask_filter_users = init_df['user_id'].isin(filter_users) 
        
        filt_df = init_df[mask_filter_items & mask_filter_users]
        filt_shp = filt_df.shape 

        if (init_shp == filt_shp):
            break 
        
        init_df = filt_df 
        init_shp = init_df.shape 
    
    return filt_df 

def user_fixed_train_test_split(df, test_size=0.2, random_state=42):

    train_data = []
    test_data = []

    for user_iid, user_data in df.groupby('user_iid'):

        user_data = user_data.sample(frac=1.0, random_state=random_state)
        train_size = int(len(user_data) * (1 - test_size))

        train_data.append(user_data.iloc[:train_size])
        test_data.append(user_data.iloc[train_size:])

    train_df = pd.concat(train_data, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_data, axis=0).reset_index(drop=True)

    return train_df, test_df 

def separate_by_gender(df, iid_to_gender):
    """Separate a data frame by female and male users.
    
    Args:
        df: pd.DataFrame
            Data frame with at least the column 'user_iid'.

        iid_to_gender: dict
            Dictionary that maps user inner index to gender.
            Keys are user inner indices, and values are "F" or "M".
            
    Returns:
        pd.DataFrame, pd.DataFrame
        - Data frame of female users.
        - Data frame of male users.
    """

    df['gender'] = df['user_iid'].map(iid_to_gender)

    pro_preds = df[df['gender'] == 'F']
    unpro_preds = df[df['gender'] != 'F']

    return pro_preds.drop(columns='gender'), unpro_preds.drop(columns='gender')

def ratio_sample_by_gender(df, ratio, iid_to_gender, random_state):
    """Count the number of rows in a data frame from female and male users 
    and sample rows to make sure of equal proportions.
    
    Args:
        df: pd.DataFrame
            Data Frame with at least the column 'user_iid'.

        ratio: float
            Ratio of data size from male users to data size from female users.
        
        iid_to_gender: dict
            Dictionary that maps user inner index to gender.
            Keys are user inner indices, and values are "F" or "M".
        
        random_state: int
            Random seed used in sampling.    
    
    Returns:
        pd.DataFrame: Sampled data frame that has the same number of 
        rows from female and male users.
    """
    pro_df, unpro_df = separate_by_gender(df, iid_to_gender)

    pro_count, unpro_count = pro_df.shape[0], unpro_df.shape[0]

    original_ratio = pro_count / unpro_count 

    if ratio >= original_ratio:
        # sample unprotected group
        unpro_df = unpro_df.sample(frac=(original_ratio/ratio), replace=False, random_state=random_state, ignore_index=True)
    else:
        # sample protected group
        pro_df = pro_df.sample(frac=(ratio/original_ratio), replace=False, random_state=random_state, ignore_index=True)
    
    return pd.concat([pro_df, unpro_df], axis=0, ignore_index=True)

def oversample_pro(df, iid_to_gender, random_state):
    """Count the number of rows in a data frame from female and male users 
    and oversample female data to match the size of male data.
    
    Args:
        df: pd.DataFrame
            Data Frame with at least the column 'user_iid'.
        
        iid_to_gender: dict
            Dictionary that maps user inner index to gender.
            Keys are user inner indices, and values are "F" or "M".
        
        random_state: int
            Random seed used in sampling.    
    
    Returns:
        pd.DataFrame: Sampled data frame that has the same number of 
        rows from female and male users.
    """
    pro_df, unpro_df = separate_by_gender(df, iid_to_gender)

    pro_count, unpro_count = pro_df.shape[0], unpro_df.shape[0]

    oversample_count = unpro_count - pro_count

    if oversample_count < 0:
        raise ValueError("Cannot oversample protected group because the number of data from" \
        "protected group is more than the number of data from unprotected group.")
    elif oversample_count == 0:
        return df
    else:    
        sampled_pro_df = pro_df.sample(n=oversample_count, replace=True, random_state=random_state, ignore_index=True)
        return pd.concat([pro_df, sampled_pro_df, unpro_df], axis=0, ignore_index=True)