import random

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp

from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

from utils import separate_by_gender

class BaseActiveLearner:

    def __init__(self, model, model_hparams, tr_df, te_df, n_users, n_items, q=10, seed=42):
        """Initialize the active learner.
        
        Parameters:
            model: tf.keras.Model
                The base recommender system used by the active learner.

                Required methods:
                    - model.score(user_ids): Return the score for all the items from each user in `user_ids`.

                    - fit(): Train the recommender system with new acquired data.
            
            tr_df: pd.DataFrame
                The training set that the active learner will query from.

                Required columns: ['user_iid', 'item_iid', 'rating']
            
            te_df: pd.DataFrame
                The test set to perform the evaluation.

                Same columns as `tr_df`.

            q: int
                The query list size. 
        """

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed 

        self.model = model
        self.model_hparams = model_hparams
        self.tr_df = tr_df
        self.te_df = te_df 
        self.n_users = n_users 
        self.n_items = n_items 
        self.q = q 

        self.col_user = 'user_iid'
        self.col_item = 'item_iid'
        self.col_rating = 'rating'
        self.col_prediction = 'rating_estimate'

        # initialize the known set with 5% of each user
        self.kn_df = self.tr_df.groupby('user_iid').sample(frac=0.05, random_state=self.seed)

        # initialize the query record (dictionary-of-keys) matrix
        # self.queried_NM = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
    
    def generate_queries(self, model):
        """Generate an array of query items for each user.
        
        Returns:
            np.ndarray
                Array of query items for each user, of shape (n_users, q).
        """
        raise NotImplementedError 

    def update_known(self, query_df):
        """Update the known set.
        
        Parameters:
            query_df: pd.DataFrame
                Queried ratings to be added to the known set.
        """
        self.kn_df = pd.concat([self.kn_df, query_df], axis=0).drop_duplicates() 

    def run_epoch(self, iid_to_gender):
        """Run an epoch of the active learning process:
            1. Re-initialize the rec sys model.
            2. Fit the rec sys model on the updated known set.
            3. Perform evaluation on the test set.
            4. Generate the items to be queried.
            5. Update the known set.
        
            Parameters:
                iid_to_gender: dict
                    A dictionary that maps `user_iid` to gender, required for Female/Male performance comparison.
            
            Returns:
                dict
                    Evaluation results for the epoch.
        """
        # re-initialize the model
        model = self.model(self.model_hparams, self.kn_df, self.te_df, self.n_users, self.n_items, self.seed)

        # re-fit the model on new known data
        model.fit()

        # evaulate this epoch
        results = self.perform_evaluation(model, top_k=10, iid_to_gender=iid_to_gender)

        # generate query items
        query_df = self.generate_queries(model)

        # update the known set
        self.update_known(query_df)

        return results

        

    def perform_evaluation(self, model, top_k, iid_to_gender):
        
        pro_te_df, unpro_te_df = separate_by_gender(self.te_df, iid_to_gender)
        
        pro_topk_scores = model.recommend_k_items(pro_te_df, top_k, remove_seen=True)
        unpro_topk_scores = model.recommend_k_items(unpro_te_df, top_k, remove_seen=True)

        params = {'k': top_k,
                  'col_user': self.col_user,
                  'col_item': self.col_item,
                  'col_prediction': self.col_prediction}
        
        pro_map = map_at_k(pro_te_df, pro_topk_scores, **params)
        unpro_map = map_at_k(unpro_te_df, unpro_topk_scores, **params)

        pro_ndcg = ndcg_at_k(pro_te_df, pro_topk_scores, **params)
        unpro_ndcg = ndcg_at_k(unpro_te_df, unpro_topk_scores, **params)

        pro_precision = precision_at_k(pro_te_df, pro_topk_scores, **params)
        unpro_precision = precision_at_k(unpro_te_df, unpro_topk_scores, **params)

        pro_recall = recall_at_k(pro_te_df, pro_topk_scores, **params)
        unpro_recall = recall_at_k(unpro_te_df, unpro_topk_scores, **params)

        results = {"map": [pro_map, unpro_map],
                  "ndcg": [pro_ndcg, unpro_ndcg],
                  "precision": [pro_precision, unpro_precision],
                  "recall": [pro_recall, unpro_recall],
                  "percentage": (self.kn_df.shape[0] / self.tr_df.shape[0])}

        return results


class RatingBasedActiveLearner(BaseActiveLearner):

    def __init__(self, strategy, model, model_hparams, tr_df, te_df, n_users, n_items, q=10, seed=42):

        super().__init__(model, model_hparams, tr_df, te_df, n_users, n_items, q, seed)
        
        avail_strategies = ['MaxRating', 'MinRating', 'MixRating', 'Random']
        if strategy not in avail_strategies:
            raise ValueError("Strategy must be one of: 'MaxRating', 'MinRating', 'MixRating', or 'Random'.")
        else:
            self.strategy = strategy 
    
    def generate_queries(self, model):
        
        user_iids_N = np.arange(self.n_users)
        
        # generate rating predictions of all users for all items,
        # and make items seen in training have predictions of `-np.inf`
        rating_preds_NM = model.score(user_iids_N, remove_seen=True)

        query_df = []

        for user_iid in user_iids_N:
            
            # user ratings
            user_tr_df = self.tr_df[self.tr_df['user_iid'] == user_iid]

            # filter the already rated items or the know ratings
            rated_bef_df = self.kn_df[self.kn_df['user_iid'] == user_iid]

            # remove the known ratings from the user ratings
            # `keep=False` means that delete all occurrences of the same row
            user_tr_df = pd.concat([user_tr_df, rated_bef_df]).drop_duplicates(ignore_index=True, keep=False)

            if user_tr_df.shape[0] == 0:
                # if we don't have any candidate ratings for this user, skip
                continue 

            # get the items to make predictions
            item_iids_M = user_tr_df['item_iid'].to_numpy()

            preds_M = [rating_preds_NM[user_iid, item_iid] for item_iid in item_iids_M]
            preds_M = np.array(preds_M)

            # assign a new column
            user_tr_df['rating_estimate'] = preds_M 

            if self.strategy == "MaxRating":
                user_tr_df_sorted = user_tr_df.sort_values(by='rating_estimate', ascending=False)
            
            elif self.strategy == "MinRating":
                user_tr_df_sorted = user_tr_df.sort_values(by='rating_estimate', ascending=True)
            
            elif self.strategy == "MixRating":
                if self.q % 2 != 0:
                    raise ValueError("To use MixRating strategy, choose a query list size (`q`) that is divisable by 2.")
                k = self.q // 2
                high_df = user_tr_df.sort_values(by='rating_estimate', ascending=False)
                low_df = user_tr_df.sort_values(by='rating_estimate', ascending=True)

                user_tr_df_sorted = pd.concat([high_df[:k], low_df[:k]])
            
            elif self.strategy == "Random":
                # random choice from the candidate pool,
                # if the size of dataframe is smaller than k, return all
                if len(user_tr_df) >= self.q:
                    user_tr_df_sorted = user_tr_df.sample(self.q)
                else:
                    user_tr_df_sorted = user_tr_df 
            
            else:
                raise ValueError("Strategy must be one of: 'MaxRating', 'MinRating', 'MixRating', or 'Random'.")
            
            query_df.append(user_tr_df_sorted.iloc[:self.q][['user_iid', 'item_iid', 'rating']])
        
        query_df = pd.concat(query_df, axis=0)

        return query_df

class NonpersonalizedActiveLearner(BaseActiveLearner):

    def __init__(self, strategy, model, model_hparams, tr_df, te_df, n_users, n_items, q=10, seed=42):

        super().__init__(model, model_hparams, tr_df, te_df, n_users, n_items, q, seed)
        
        avail_strategies = ['pop', 'var', 'popvar', 'ge', 'ran']
        if strategy not in avail_strategies:
            raise ValueError("Strategy must be one of: 'pop', 'var', 'popvar', 'ge', or 'ran'.")
        else:
            self.strategy = strategy 
    
    def generate_queries(self, model):
        pass

    def _generate_queries_pop(self):
        """Popularity: select the most popular items in the candidate set."""
        
        item_pop_df = (self.tr_df.groupby('item_iid')['rating'].count()
                       .reset_index(name='pop').sort_values(by='pop', ascending=False, ignore_index=True))
        
        min_pop, max_pop = item_pop_df['pop'].iloc[[-1, 0]]
        item_pop_df['norm_pop'] = (item_pop_df['pop'] - min_pop) / (max_pop - min_pop)
        
        return item_pop_df
    
    def _generate_queries_var(self):
        """Variance: select the items with highest variance."""

        item_var_df = (self.tr_df.groupby('item_iid')['rating'].var(ddof=0)
                       .reset_index(name='var').sort_values(by='var', ascending=False, ignore_index=True))
        
        return item_var_df 
    
    def _generate_queries_popvar(self):

        """Popularity * Variance: normalized popularity times variance."""
        item_pop_df = self._generate_queries_pop()
        min_pop, max_pop = item_pop_df['pop'].iloc[[-1, 0]]
        item_pop_df['norm_pop'] = (item_pop_df['pop'] - min_pop) / (max_pop - min_pop)

        item_var_df = self._generate_queries_var()

        item_popvar_df = pd.merge(item_pop_df, item_var_df, on='item_iid')
        item_popvar_df['pop_var'] = np.sqrt(item_popvar_df['norm_pop']) * np.asarray(item_popvar_df['var'])

        item_popvar_df = item_popvar_df.sort_values(by='pop_var', ascending=False, ignore_index=True)

        return item_popvar_df 



        

    
