import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

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
    """Split a data frame into train and test sets, using 20% of each user's data for test.
    
    Args:
        df: pd.DataFrame
            Data frame with at least the column 'user_iid'.
        test_size: float
            Proportion of data to allocate to the test set.
        random_state: int
            Random seed for reproducibility.
    
    Returns:
        pd.DataFrame, pd.DataFrame
        - Train data frame.
        - Test data frame.
    """

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

def compute_rmse(model, te_df):
    """Compute Root Mean Squared Error (RMSE) for a test dataframe using a trained LightGCNRecommender model.
    
    Args:
        model: LightGCNRecommender
            A trained LightGCNRecommender model from LightGCN.py
        te_df: pd.DataFrame
            Test dataframe with columns ['user_iid', 'item_iid', 'rating']
            - user_iid: user inner ID (integer index)
            - item_iid: item inner ID (integer index)  
            - rating: actual rating value
    
    Returns:
        float: Root Mean Squared Error between predicted and actual ratings
    """
    # Get unique user IDs from test set
    user_ids = te_df['user_iid'].unique()
    
    # Get predicted scores for all users and items
    # remove_seen=False ensures we get actual predictions for all items (including seen ones)
    predicted_scores = np.clip(model.score(user_ids, remove_seen=False), 1, 5)
    
    # Create a mapping from user_iid to index in predicted_scores matrix
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    
    # Extract predicted scores for each (user_iid, item_iid) pair in te_df
    predictions = []
    actuals = []
    
    for _, row in te_df.iterrows():
        user_iid = row['user_iid']
        item_iid = row['item_iid']
        actual_rating = row['rating']
        
        # Get the predicted score from the score matrix
        user_idx = user_id_to_idx[user_iid]
        predicted_rating = predicted_scores[user_idx, item_iid]
        
        predictions.append(predicted_rating)
        actuals.append(actual_rating)
    
    # Compute RMSE
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return rmse

def compute_rocauc(model, te_df):
    """Compute ROC AUC score for a test dataframe using a trained LightGCNRecommender model.
    
    This function evaluates the model's ranking ability by:
    1. For each user, identifying positive items (items in test set) vs negative items (unseen items)
    2. Excluding items seen during training from evaluation
    3. Getting predicted scores for all items
    4. Computing AUC for each user (how well positives are ranked above negatives)
    5. Averaging AUC scores across all users
    
    Args:
        model: LightGCNRecommender
            A trained LightGCNRecommender model from LightGCN.py
        te_df: pd.DataFrame
            Test dataframe with columns ['user_iid', 'item_iid', 'rating']
            - user_iid: user inner ID (integer index)
            - item_iid: item inner ID (integer index)  
            - rating: actual rating value
    
    Returns:
        float: Average ROC AUC score across all users (range: 0.0 to 1.0)
    """
    # Get unique user IDs from test set
    user_ids = te_df['user_iid'].unique()
    
    # Get predicted scores with training items masked as -inf
    # Since train-test split ensures no overlap, test items won't be -inf
    predicted_scores = model.score(user_ids, remove_seen=True)
    
    # Create a mapping from user_iid to index in predicted_scores matrix
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    
    # Group test items by user
    user_to_test_items = te_df.groupby('user_iid')['item_iid'].apply(set).to_dict()
    
    # Compute AUC for each user
    auc_scores = []
    
    for user_iid in user_ids:
        # Get positive items for this user (items in test set)
        if user_iid not in user_to_test_items:
            continue
        
        positive_items = user_to_test_items[user_iid]
        user_idx = user_id_to_idx[user_iid]
        
        # Identify items to exclude (training items marked as -inf)
        user_scores = predicted_scores[user_idx, :]
        train_item_mask = np.isinf(user_scores) & (user_scores < 0)
        
        # Create evaluation mask: only evaluate items not in training
        eval_mask = ~train_item_mask
        eval_item_indices = np.where(eval_mask)[0]
        
        # Skip if no items to evaluate or no positive items
        if len(eval_item_indices) == 0 or len(positive_items) == 0:
            continue
        
        # Skip if all evaluated items are positive (no negatives to compare)
        if len(positive_items) == len(eval_item_indices):
            continue
        
        # Create binary labels for evaluated items only
        # 1 for items in test set, 0 for unseen items (neither in training nor test)
        labels = np.array([1 if item in positive_items else 0 
                          for item in eval_item_indices])
        
        # Get predicted scores for evaluated items (excluding training items with -inf)
        user_predictions = user_scores[eval_item_indices] >= 3
        
        # Compute AUC for this user
        try:
            auc = roc_auc_score(labels, user_predictions)
            auc_scores.append(auc)
        except ValueError:
            # Skip if all labels are the same class
            continue
    
    # Return average AUC across all users
    if len(auc_scores) == 0:
        return 0.0
    
    return np.mean(auc_scores)