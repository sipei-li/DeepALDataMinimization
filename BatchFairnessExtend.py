import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

from utils import separate_by_gender
from ActiveLearner import BaseActiveLearner


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

    def generate_queries(self, model):
        """Generate queries by selecting items based on current fairness ordering.
        
        This method recalculates the fairness ordering each time it's called,
        ensuring that the active learner adapts to the current model state.
        """
        # Recalculate fairness ordering based on current model
        iter_df = self._generate_iters_fairness(model, k=20)
        sorted_fairness_items = iter_df['item_iid'].tolist()
        
        user_iids_N = np.arange(self.n_users)
        query_df = []
        
        for user_iid in user_iids_N:
            # items queried before for this user
            queried_bef_items = np.argwhere(self.queried_NM[user_iid, :] > 0)[:, 1]
            
            if len(queried_bef_items) == self.n_items:
                continue
            
            # Find items in fairness order that haven't been queried for this user
            available_items = [item_iid for item_iid in sorted_fairness_items 
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

    def _generate_iters_fairness(self, current_model, k):
        """Batch-based Fairness Extend.
        
        Instead of calculating fairness gain for each item individually, this method
        processes items in batches of size self.q. Items within each batch are 
        randomly sampled and assigned the same fairness gain score.
        
        Args:
            current_model: fitted model
                The currently fitted model to use as baseline.
            k: int
                Number of top k items when calculating metrics.
            
        Returns:
            pd.DataFrame
                Dataframe with columns ['item_iid', 'fairness_wo', 'fairness_gain'], 
                sorted by 'fairness_gain' in descending order.
        """
        
        # Use the current model as baseline
        current_top_k_scores = current_model.recommend_k_items(self.te_df, k, remove_seen=True)
        current_fairness_gap = self._calculate_fairness_gap(self.te_df, current_top_k_scores, k)

        def retrain_without_batch(item_batch):
            """Retrain model without a batch of items and calculate fairness gap."""
            # Remove all items in the batch from current known data
            kn_filt = self.kn_df[~self.kn_df['item_iid'].isin(item_batch)]
            
            # Check if filtered data is sufficient for training
            if kn_filt.empty or len(kn_filt) < 10:  # Minimum threshold for meaningful training
                # Return current fairness gap if no meaningful data remains
                return current_fairness_gap
            
            # Check if we have interactions for at least some users
            unique_users = kn_filt['user_iid'].nunique()
            if unique_users < 2:  # Need at least 2 users for meaningful training
                return current_fairness_gap
            
            # Create a new model instance with isolated TensorFlow state
            # Use a unique seed based on the batch to ensure reproducibility
            batch_seed = self.seed + hash(tuple(sorted(item_batch))) % 10000
            temp_model = self.model(self.model_hparams, kn_filt, self.te_df, self.n_users, self.n_items, batch_seed)
            temp_model.fit()

            temp_top_k_scores = temp_model.recommend_k_items(self.te_df, k, remove_seen=True)
            temp_fairness_gap = self._calculate_fairness_gap(self.te_df, temp_top_k_scores, k)
            
            return temp_fairness_gap

        # Get items that exist in the current known set
        # known_items = list(self.kn_df['item_iid'].unique())
        # All items
        all_items = list(range(self.n_items))
        
        # Shuffle items to introduce randomness
        # random.shuffle(known_items)
        random.shuffle(all_items)
        
        # Create batches of size self.q
        # batches = []
        # for i in range(0, len(known_items), self.q):
        #     batch = known_items[i:i + self.q]
        #     batches.append(batch)
        batches = []
        for i in range(0, len(all_items), self.q):
            batch = all_items[i:i + self.q]
            batches.append(batch)
        
        # Calculate fairness gain for each batch
        results = []
        for batch_idx, item_batch in enumerate(batches):
            # Calculate fairness gap when this batch is removed
            batch_fairness_gap = retrain_without_batch(item_batch)
            batch_fairness_gain = current_fairness_gap - batch_fairness_gap
            
            # Assign the same fairness gain to all items in the batch
            # but add small random noise to create ordering within batch
            for item_iid in item_batch:
                # Add small random noise to break ties within batch
                noise = random.uniform(-0.001, 0.001)
                results.append([item_iid, batch_fairness_gap, batch_fairness_gain + noise])
        
        item_fairness_df = pd.DataFrame(results, columns=['item_iid', 'fairness_wo', 'fairness_gain'])
        
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
