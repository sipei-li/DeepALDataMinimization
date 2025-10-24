import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import multiprocessing as mp

from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

from utils import separate_by_gender
from ActiveLearner import BaseActiveLearner


# Module-level function for multiprocessing (must be picklable)
def _worker_retrain_without_item(args):
    """Worker function for multiprocessing that retrains model without a specific item.
    
    Args:
        args: tuple
            (item_iid, model_class, model_hparams, tr_df, te_df, n_users, n_items, seed, 
             oracle_k, fairness_metric, iid_to_gender, col_user, col_item, col_prediction)
    
    Returns:
        list: [item_iid, fairness_gap_without_item]
    """
    (item_iid, model_class, model_hparams, tr_df, te_df, n_users, n_items, seed, 
     oracle_k, fairness_metric, iid_to_gender, col_user, col_item, col_prediction) = args
    
    # Remove all ratings for this specific item from training data
    tr_filt = tr_df[tr_df['item_iid'] != item_iid]
    
    # Create a new model instance
    temp_model = model_class(model_hparams, tr_filt, te_df, n_users, n_items, seed)
    temp_model.fit()
    
    temp_top_k_scores = temp_model.recommend_k_items(te_df, oracle_k, remove_seen=True)
    
    # Calculate fairness gap
    pro_te_df, unpro_te_df = separate_by_gender(te_df, iid_to_gender)
    pro_topk_scores, unpro_topk_scores = separate_by_gender(temp_top_k_scores, iid_to_gender)
    
    params = {
        'k': oracle_k,
        'col_user': col_user,
        'col_item': col_item,
        'col_prediction': col_prediction
    }
    
    # Calculate the specified metric for both groups
    if fairness_metric == 'precision':
        pro_metric = precision_at_k(pro_te_df, pro_topk_scores, **params)
        unpro_metric = precision_at_k(unpro_te_df, unpro_topk_scores, **params)
    elif fairness_metric == 'recall':
        pro_metric = recall_at_k(pro_te_df, pro_topk_scores, **params)
        unpro_metric = recall_at_k(unpro_te_df, unpro_topk_scores, **params)
    elif fairness_metric == 'ndcg':
        pro_metric = ndcg_at_k(pro_te_df, pro_topk_scores, **params)
        unpro_metric = ndcg_at_k(unpro_te_df, unpro_topk_scores, **params)
    
    temp_fairness_gap = abs(pro_metric - unpro_metric)
    
    return [item_iid, temp_fairness_gap]


class FairnessExtendActiveLearner(BaseActiveLearner):
    """Batch Fairness Extend Active Learner.
    
    This active learner uses a fairness-based greedy extend strategy, sorting items based on 
    fairness gain. Fairness is measured by the difference in metrics (precision@k, recall@k,
    or NDCG@k) between male and female users.
    """

    def __init__(self, fairness_metric, model, model_hparams, tr_df, te_df, n_users, n_items, iid_to_gender, q=10, seed=42):
        """Initialize the FairnessExtend Active Learner.
        
        Args:
            fairness_metric: str
                The metric to use for fairness calculation. Must be one of:
                'precision', 'recall', 'ndcg'
            model: tf.keras.Model
                The base recommender system used by the active learner.
            model_hparams: dict
                Hyperparameters for the model.
            tr_df: pd.DataFrame
                The training set that the active learner will query from.
            te_df: pd.DataFrame
                The test set to perform the evaluation.
            n_users: int
                Number of users in the dataset.
            n_items: int
                Number of items in the dataset.
            iid_to_gender: dict
                Dictionary that maps user inner index to gender.
            q: int
                The query list size.
            seed: int
                Random seed.
        """
        super().__init__(model, model_hparams, tr_df, te_df, n_users, n_items, iid_to_gender, q, seed)
        
        avail_metrics = ['precision', 'recall', 'ndcg']
        if fairness_metric not in avail_metrics:
            raise ValueError(f"Fairness metric must be one of: {avail_metrics}")
        else:
            self.fairness_metric = fairness_metric
        
        # initialize the query record (dictionary-of-keys) matrix
        self.queried_NM = sp.dok_matrix((self.n_users, self.n_items), dtype=np.int32)

        # update the query record matrix with the initial known set
        for _, row in self.kn_df.iterrows():
            user_iid, item_iid = row[['user_iid', 'item_iid']]
            self.queried_NM[user_iid, item_iid] = 1
        
        # Initialize item ordering based on fairness (computed once on tr_df)
        self.initialize_iters()
    
    def initialize_iters(self):
        """Initialize the item ordering based on fairness gain.
        
        This method computes the fairness ordering once on the full training set,
        similar to the greedy extend oracle approach.
        """
        iter_df = self._generate_iters_fairness(k=20)
        self.sorted_i = iter_df['item_iid'].tolist()
        self.i_indx = 0

    def generate_queries(self, model):
        """Generate queries by selecting items based on pre-computed fairness ordering.
        
        This method uses a pre-computed fairness ordering (calculated once on tr_df)
        and selects items for each user from this fixed ordering.
        """
        user_iids_N = np.arange(self.n_users)
        query_df = []
        
        for user_iid in user_iids_N:
            # items queried before for this user
            queried_bef_items = np.argwhere(self.queried_NM[user_iid, :] > 0)[:, 1]
            
            if len(queried_bef_items) == self.n_items:
                continue
            
            # Find items in fairness order that haven't been queried for this user
            available_items = [item_iid for item_iid in self.sorted_i 
                             if item_iid not in queried_bef_items]
            
            if not available_items:
                continue
                
            # Select top q items based on fairness ordering
            query_items = available_items[:self.q]
            
            # update query record matrix
            for item_iid in query_items:
                self.queried_NM[user_iid, item_iid] = 1
            
            # user training set
            user_tr_df = self.tr_df[self.tr_df['user_iid'] == user_iid]
            
            # query items from training set that are in the selected items
            user_query_df = user_tr_df[user_tr_df['item_iid'].isin(query_items)]
            
            if not user_query_df.empty:
                query_df.append(user_query_df[['user_iid', 'item_iid', 'rating']])
        
        if query_df:
            return pd.concat(query_df, axis=0)
        else:
            # Return empty dataframe with correct columns if no queries available
            return pd.DataFrame(columns=['user_iid', 'item_iid', 'rating'])

    def _generate_iters_fairness(self, k):
        """Item-based Fairness Extend (Oracle approach).
        
        Calculates fairness gain for each item by training an oracle model on the full
        training set, then retraining without each item and comparing fairness gaps.
        This is computed once during initialization.
        
        Args:
            k: int
                Number of top k items when calculating metrics.
            
        Returns:
            pd.DataFrame
                Dataframe with columns ['item_iid', 'fairness_wo', 'fairness_gain'], 
                sorted by 'fairness_gain' in descending order.
        """
        
        # Fit the oracle model on the whole training set
        oracle = self.model(self.model_hparams, self.tr_df, self.te_df, self.n_users, self.n_items, self.seed)
        oracle.fit()
        oracle_top_k_scores = oracle.recommend_k_items(self.te_df, k, remove_seen=True)
        oracle_fairness_gap = self._calculate_fairness_gap(self.te_df, oracle_top_k_scores, k)

        # Prepare arguments for each worker process
        worker_args = [
            (item_iid, self.model, self.model_hparams, self.tr_df, self.te_df, 
             self.n_users, self.n_items, self.seed, k, self.fairness_metric, 
             self.iid_to_gender, self.col_user, self.col_item, self.col_prediction)
            for item_iid in range(self.n_items)
        ]
        
        # Calculate fairness gain for all items using multiprocessing
        with mp.Pool(processes=25) as pool:
            results = pool.map(_worker_retrain_without_item, worker_args)
        
        item_fairness_df = pd.DataFrame(results, columns=['item_iid', 'fairness_wo'])
        item_fairness_df['fairness_gain'] = oracle_fairness_gap - item_fairness_df['fairness_wo']
        
        # Sort by fairness gain (higher gain = more important to include)
        item_fairness_df = item_fairness_df.sort_values(by='fairness_gain', ascending=False, ignore_index=True)

        return item_fairness_df

    def _calculate_fairness_gap(self, te_df, top_k_scores, k):
        """Calculate the fairness gap between male and female users.
        
        Args:
            te_df: pd.DataFrame
                Test dataframe.
            top_k_scores: pd.DataFrame
                Top-k recommendation scores.
            k: int
                Number of top items.
                
        Returns:
            float
                Absolute difference in the specified metric between male and female users.
        """
        # Separate test data by gender
        pro_te_df, unpro_te_df = separate_by_gender(te_df, self.iid_to_gender)
        
        # Separate recommendations by gender
        pro_topk_scores, unpro_topk_scores = separate_by_gender(top_k_scores, self.iid_to_gender)
        
        params = {
            'k': k,
            'col_user': self.col_user,
            'col_item': self.col_item,
            'col_prediction': self.col_prediction
        }
        
        # Calculate the specified metric for both groups
        if self.fairness_metric == 'precision':
            pro_metric = precision_at_k(pro_te_df, pro_topk_scores, **params)
            unpro_metric = precision_at_k(unpro_te_df, unpro_topk_scores, **params)
        elif self.fairness_metric == 'recall':
            pro_metric = recall_at_k(pro_te_df, pro_topk_scores, **params)
            unpro_metric = recall_at_k(unpro_te_df, unpro_topk_scores, **params)
        elif self.fairness_metric == 'ndcg':
            pro_metric = ndcg_at_k(pro_te_df, pro_topk_scores, **params)
            unpro_metric = ndcg_at_k(unpro_te_df, unpro_topk_scores, **params)
        
        # Return the absolute difference (fairness gap)
        return abs(pro_metric - unpro_metric)
