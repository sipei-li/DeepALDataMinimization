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

from utils import separate_by_gender, ratio_sample_by_gender, oversample_pro

class BaseActiveLearner:

    def __init__(self, model, model_hparams, tr_df, te_df, n_users, n_items, iid_to_gender, q=10, seed=42, oversample=False, ratio=None):
        """Initialize the active learner.
        
        Args:
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
            
            oversample: bool
                Whether we are doing oversampling for female users.

            ratio: float
                Ratio of number of ratings collected at each epoch from female and male users.
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
        self.iid_to_gender = iid_to_gender
        self.q = q 

        # experiment settings
        self.oversample = oversample
        self.ratio = ratio
        
        self.col_user = 'user_iid'
        self.col_item = 'item_iid'
        self.col_rating = 'rating'
        self.col_prediction = 'rating_estimate'

        # initialize the known set with 5% of each user
        self.kn_df = self.tr_df.groupby('user_iid').sample(frac=0.05, random_state=self.seed)

    
    def generate_queries(self, model):
        """Generate an array of query items for each user.
        
        Returns:
            np.ndarray
                Array of query items for each user, of shape (n_users, q).
        """
        raise NotImplementedError 

    def update_known(self, query_df):
        """Update the known set.
        
        Args:
            query_df: pd.DataFrame
                Queried ratings to be added to the known set.
        """
        self.kn_df = pd.concat([self.kn_df, query_df], axis=0).drop_duplicates() 

    def run_epoch(self):
        """Run an epoch of the active learning process:
            1. Re-initialize the rec sys model.
            2. Fit the rec sys model on the updated known set.
            3. Perform evaluation on the test set.
            4. Generate the items to be queried.
            5. Update the known set.
        
            Args:
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
        results = self.perform_evaluation(model, top_k=10)

        # generate query items
        query_df = self.generate_queries(model)

        # update the known set
        self.update_known(query_df)

        return results

        

    def perform_evaluation(self, model, top_k):
        
        pro_te_df, unpro_te_df = separate_by_gender(self.te_df, self.iid_to_gender)
        
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

        pro_kn_df, unpro_kn_df = separate_by_gender(self.kn_df, self.iid_to_gender)
        pro_count = pro_kn_df.shape[0]
        unpro_count = unpro_kn_df.shape[0]

        results = {"map_pro": pro_map,
                   "map_unpro": unpro_map,
                   "ndcg_pro": pro_ndcg,
                   "ndcg_unpro": unpro_ndcg,
                   "precision_pro": pro_precision,
                   "precision_unpro": unpro_precision,
                   "recall_pro": pro_recall,
                   "recall_unpro": unpro_recall,
                   "count_pro": pro_count,
                   "count_unpro": unpro_count,
                   "percentage": (self.kn_df.shape[0]/ self.tr_df.shape[0]),
        }

        return results


class RatingBasedActiveLearner(BaseActiveLearner):

    def __init__(
        self, 
        strategy, 
        model, 
        model_hparams, 
        tr_df, 
        te_df, 
        n_users, 
        n_items, 
        iid_to_gender, 
        q=10, 
        seed=42, 
        oversample=False, 
        ratio=None
    ):

        super().__init__(model, model_hparams, tr_df, te_df, n_users, n_items, iid_to_gender, q, seed, oversample, ratio)
        
        avail_strategies = ['MaxRating', 'MinRating', 'MixRating', 'Random']
        if strategy not in avail_strategies:
            raise ValueError("Strategy must be one of:", avail_strategies)
        else:
            self.strategy = strategy 
        
        # initialize the query record (dictionary-of-keys) matrix
        self.queried_NM = sp.dok_matrix((self.n_users, self.n_items), dtype=np.int32)

        # update the query record matrix with the initial known set
        for _, row in self.kn_df.iterrows():
            user_iid, item_iid = row[['user_iid', 'item_iid']]
            self.queried_NM[user_iid, item_iid] = 1
        
        if self.oversample:
            # oversample the female data in the initial known set
            self.kn_df = oversample_pro(self.kn_df, self.iid_to_gender, random_state=self.seed)
        elif self.ratio:
            # ratio sample the initial known set
            self.kn_df = ratio_sample_by_gender(self.kn_df, self.ratio, self.iid_to_gender, random_state=self.seed)

    def generate_queries(self, model):
        
        user_iids_N = np.arange(self.n_users)
        
        # generate rating predictions of all users for all items,
        # and make items seen in training have predictions of `-np.inf`
        rating_preds_NM = model.score(user_iids_N, remove_seen=True)

        query_df = []

        for user_iid in user_iids_N:

            # items queried before
            queried_bef_items = np.argwhere(self.queried_NM[user_iid, :] > 0)[:, 1]
            
            if len(queried_bef_items) == self.n_items:
                continue

            # candidate set - all items
            user_df = pd.DataFrame(
                {
                    'item_iid': np.arange(self.n_items),
                    'rating_estimate': rating_preds_NM[user_iid, :],
                }
            )
            
            # remove the items that have been queried before from candidate set
            user_df = user_df[~user_df['item_iid'].isin(queried_bef_items)]

            if self.strategy == "MaxRating":
                user_df_sorted = user_df.sort_values(by='rating_estimate', ascending=False)
            
            elif self.strategy == "MinRating":
                user_df_sorted = user_df.sort_values(by='rating_estimate', ascending=True)
            
            elif self.strategy == "MixRating":
                if self.q % 2 != 0:
                    raise ValueError("To use MixRating strategy, choose a query list size (`q`) that is divisable by 2.")
                k = self.q // 2
                high_df = user_df.sort_values(by='rating_estimate', ascending=False)
                low_df = user_df.sort_values(by='rating_estimate', ascending=True)

                user_df_sorted = pd.concat([high_df[:k], low_df[:k]])
            
            elif self.strategy == "Random":
                # random choice from the candidate pool,
                # if the size of dataframe is smaller than k, return all
                if len(user_df) >= self.q:
                    user_df_sorted = user_df.sample(self.q)
                else:
                    user_df_sorted = user_df 
            
            else:
                raise ValueError("Strategy must be one of: 'MaxRating', 'MinRating', 'MixRating', or 'Random'.")
            
            # keep the top q items
            query_items = user_df_sorted.iloc[:self.q]['item_iid'].to_list()
            
            # update query record matrix
            for item_iid in query_items:
                self.queried_NM[user_iid, item_iid] = 1
            
            # user training set
            user_tr_df = self.tr_df[self.tr_df['user_iid'] == user_iid]

            # query items from training set that are in the top q
            user_query_df = user_tr_df[user_tr_df['item_iid'].isin(query_items)]
            
            query_df.append(user_query_df[['user_iid', 'item_iid', 'rating']])
        
        query_df = pd.concat(query_df, axis=0)

        if self.oversample:
            query_df = oversample_pro(query_df, self.iid_to_gender, random_state=self.seed)
        elif self.ratio:
            query_df = ratio_sample_by_gender(query_df, self.ratio, self.iid_to_gender, random_state=self.seed)

        return query_df

class NonpersonalizedActiveLearner(BaseActiveLearner):

    def __init__(
        self, 
        strategy, 
        model, 
        model_hparams, 
        tr_df, 
        te_df, 
        n_users, 
        n_items, 
        iid_to_gender, 
        q=10, 
        seed=42,
        oversample=False,
        ratio=None
    ):

        super().__init__(model, model_hparams, tr_df, te_df, n_users, n_items, iid_to_gender, q, seed, oversample, ratio)
        
        avail_strategies = ['pop', 'var', 'popvar', 'ge', 'ran']
        if strategy not in avail_strategies:
            raise ValueError("Strategy must be one of:", avail_strategies)
        else:
            self.strategy = strategy 
        
        self.initialize_iters()
    
    def generate_queries(self, model):
        chosen_is = self.sorted_i[self.i_indx:self.i_indx+self.q]
        query_df = self.tr_df[self.tr_df['item_iid'].isin(chosen_is)]

        self.i_indx += self.q

        if self.oversample:
            query_df = oversample_pro(query_df, self.iid_to_gender, random_state=self.seed)
        elif self.ratio:
            query_df = ratio_sample_by_gender(query_df, self.ratio, self.iid_to_gender, random_state=self.seed)

        return query_df
    
    def initialize_iters(self):
        if self.strategy == 'ran':
            self.sorted_i = random.sample(range(self.n_items), self.n_items)
        else:
            if self.strategy == 'pop':
                iter_df = self._generate_iters_pop()
            elif self.strategy == 'var':
                iter_df = self._generate_iters_var()
            elif self.strategy == 'popvar':
                iter_df = self._generate_iters_popvar()
            elif self.strategy == 'ge':
                iter_df = self._generate_iters_ge(k=10)

            self.sorted_i = iter_df['item_iid'].tolist()
        
        self.i_indx = 0
        
    def _generate_iters_pop(self):
        """Popularity: select the most popular items in the candidate set."""
        
        item_pop_df = (self.tr_df.groupby('item_iid')['rating'].count()
                       .reset_index(name='pop').sort_values(by='pop', ascending=False, ignore_index=True))
        
        min_pop, max_pop = item_pop_df['pop'].iloc[[-1, 0]]
        item_pop_df['norm_pop'] = (item_pop_df['pop'] - min_pop) / (max_pop - min_pop)
        
        return item_pop_df
    
    def _generate_iters_var(self):
        """Variance: select the items with highest variance."""

        item_var_df = (self.tr_df.groupby('item_iid')['rating'].var(ddof=0)
                       .reset_index(name='var').sort_values(by='var', ascending=False, ignore_index=True))
        
        return item_var_df 
    
    def _generate_iters_popvar(self):

        """Popularity * Variance: normalized popularity times variance."""
        item_pop_df = self._generate_queries_pop()
        min_pop, max_pop = item_pop_df['pop'].iloc[[-1, 0]]
        item_pop_df['norm_pop'] = (item_pop_df['pop'] - min_pop) / (max_pop - min_pop)

        item_var_df = self._generate_queries_var()

        item_popvar_df = pd.merge(item_pop_df, item_var_df, on='item_iid')
        item_popvar_df['pop_var'] = np.sqrt(item_popvar_df['norm_pop']) * np.asarray(item_popvar_df['var'])

        item_popvar_df = item_popvar_df.sort_values(by='pop_var', ascending=False, ignore_index=True)

        return item_popvar_df 
    
    def _generate_iters_ge(self, k):
        """Greedy Extend.
        
        Args:
            k: int
                Number of top k items when calculating precision@k.
            
        Returns:
            pd.DataFrame
                Dataframe with columns ['item_iid', 'precision_wo', 'precision_gain'], 
                sorted by 'precision_gain'.
        """
        
        # fit the oracle on the whole training set
        oracle = self.model(self.model_hparams, self.tr_df, self.te_df, self.n_users, self.n_items, self.seed)
        oracle.fit()
        oracle_top_k_scores = oracle.recommend_k_items(self.te_df, 10, remove_seen=True)
        oracle_precision_at_k = precision_at_k(self.te_df,
                                               oracle_top_k_scores,
                                               k=10,
                                               col_user=self.col_user,
                                               col_item=self.col_item,
                                               col_prediction=self.col_prediction
                                               )

        def retrain_without(item_iid):
            tr_filt = self.tr_df[self.tr_df['item_iid'] != item_iid]
            temp_model = self.model(self.model_hparams, tr_filt, self.te_df, self.n_users, self.n_items, self.seed)
            temp_model.fit()

            temp_top_k_scores = temp_model.recommend_k_items(self.te_df, 10, remove_seen=True)
            
            # we are using precision at k for calculating the gain for each item
            # should we use a different metric?
            temp_precision_at_k = precision_at_k(self.te_df, 
                                                 temp_top_k_scores, 
                                                 k=10, 
                                                 col_user=self.col_user, 
                                                 col_item=self.col_item,
                                                 col_prediction=self.col_prediction
                                                 )
            return [item_iid, temp_precision_at_k]

        results = []
        for item_iid in range(self.n_items):
            results.append(retrain_without(item_iid))
        
        item_ge_df = pd.DataFrame(results, columns=['item_iid', 'precision_wo'])
        item_ge_df['precision_gain'] = oracle_precision_at_k - item_ge_df['precision_wo']
        item_ge_df = item_ge_df.sort_values(by='precision_gain', ascending=False, ignore_index=True)

        return item_ge_df 


    
class kNNActiveLearner(BaseActiveLearner):

    def __init__(self, 
                 model, 
                 model_hparams, 
                 tr_df, 
                 te_df, 
                 n_users, 
                 n_items, 
                 iid_to_gender, 
                 q=10, 
                 seed=42):
        super().__init__(model, model_hparams, tr_df, te_df, n_users, n_items, iid_to_gender, q, seed)

    def generate_queries(self, model):
        
        user_iids_N = np.arange(self.n_users)

        query_df = []

        for user_iid in user_iids_N:

            # user ratings - candidate set
            user_tr_df = self.tr_df[self.tr_df['user_iid'] == user_iid]

            # filter the already rated items or the known ratings
            rated_bef_df = self.kn_df[self.kn_df['user_iid'] == user_iid]

            # remove the known ratings from the candidate set
            # `keep=False` means that we delete all occurrences of the same row
            user_tr_df = pd.concat([user_tr_df, rated_bef_df]).drop_duplicates(ignore_index=True, keep=False)

            if user_tr_df.shape[0] == 0:
                # if we don't have any candidate ratings left for this user, skip
                continue 

    def update_item_sim_mat(self):
        item_user_mat = self.tr_df
        # TODO
