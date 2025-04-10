import os
import random 

import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.sparse as sp

from utils import get_top_k_scored_items
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class LightGCNRecommender(tf.keras.Model):

    def __init__(self, hparams, tr_df, te_df, n_users, n_items, seed=None):
        """Initialize the model.
        
        Args:
            hparams: dict, a dictionary that holds the entire set of hyperparameters
            tr_df: pandas.DataFrame, training set dataframe of at least 3 columns: ['user_iid', 'item_iid', 'rating']
            te_df: pandas.DataFrame, training set dataframe of at least 3 columns: ['user_iid', 'item_iid', 'rating']
            n_users: int, number of users in the dataset
            n_items: int, number of items in the dataset
            seed: int, random seed
        """
        
        super(LightGCNRecommender, self).__init__()
        
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed 

        self.tr_df = tr_df
        self.te_df = te_df
        self.n_users = n_users
        self.n_items = n_items 

        self.epochs = hparams['epochs']
        self.lr = hparams['learning_rate']
        self.emb_dim = hparams['embed_size']
        self.batch_size = hparams['batch_size']
        self.n_layers = hparams['n_layers']
        self.decay = hparams['decay']
        self.eval_epoch = hparams['eval_epoch']
        self.top_k = hparams['top_k']
        # self.save_model = hparams.save_model
        # self.save_epoch = hparams.save_epoch
        # self.metrics = hparams['metrics']
        # self.model_dir = hparams.MODEL_DIR

        self.col_user = 'user_iid'
        self.col_item = 'item_iid'
        self.col_rating = 'rating'
        self.col_prediction = 'rating_estimate'


        self._init_train_data()
        self.norm_adj = self.create_norm_adj_mat()
        self.A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        self.users = tf.keras.layers.Input(shape=(), dtype=tf.int32)
        self.pos_items = tf.keras.layers.Input(shape=(), dtype=tf.int32)
        self.neg_items = tf.keras.layers.Input(shape=(), dtype=tf.int32)

        self._weights = self._init_weights()

        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)

    def _init_train_data(self):
        """Record items interacted with each user in a dataframe self.interact_status, and
        create adjacency matrix self.R."""

        self.interact_status = (
            self.tr_df.groupby(self.col_user)[self.col_item]
            .apply(set)
            .reset_index()
            .rename(columns={self.col_item: self.col_item + "_interacted"})
        )

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R[self.tr_df[self.col_user], self.tr_df[self.col_item]] = 1.0

    def create_norm_adj_mat(self):
        """
        Create normalized adjacency matrix.
        
        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.
        """
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R 
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)

        return norm_adj_mat.tocsr()

    
    def _init_weights(self):
        """Initialize user and item embeddings.
        
        Returns:
            dict: a dictionary of embeddings of all users and items, 
                with keys `user_embedding` and `item_embedding`
        """
        
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=self.seed)
        return {
            "user_embedding": tf.Variable(initializer([self.n_users, self.emb_dim]), name="user_embedding"),
            "item_embedding": tf.Variable(initializer([self.n_items, self.emb_dim]), name="item_embedding")
        }
    
    def _create_lightgcn_embed(self):
        """Calculate the average embeddings of users and items after every layer of the model.
        
        Returns:
            tf.Tensor, tf.Tensor: 
            - average user embeddings
            - average item embeddings
        """
        # A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        ego_embeddings = tf.concat([self._weights["user_embedding"], self._weights["item_embedding"]], axis=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            # embedding propagation
            ego_embeddings = tf.sparse.sparse_dense_matmul(self.A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]
        
        all_embeddings = tf.reduce_mean(tf.stack(all_embeddings, axis=1), axis=1, keepdims=False)

        # separate user and item embeddings
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], axis=0)
        return u_g_embeddings, i_g_embeddings
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert a scipy sparse matrix to tf.SparseTensor.
        
        Returns:
            tf.SparseTensor: SparseTensor after conversion.
        """
        
        coo = X.tocoo().astype(np.float32) # convert to coordinate format: [(r,c,v)]
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.sparse.SparseTensor(indices, coo.data, coo.shape)
    
    def train_loader(self, batch_size):
        """
        Sample train data for every batch. One positive item and one negative item sampled for each user.

        Args:
            batch_size: int, Batch size of users.
        
        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray:
            - Sampled users.
            - Sampled positive items.
            - Sampled negative items.
        """

        def sample_neg(x):
            if len(x) >= self.n_items:
                raise ValueError("A user has voted in every item. Can't find a negative sample.")
            while True:
                neg_id = random.randint(0, self.n_items - 1)
                if neg_id not in x:
                    return neg_id  

        indices = range(self.n_users)
        if self.n_users < batch_size:
            users = [random.choice(indices) for _ in range(batch_size)]
        else:
            users = random.sample(indices, batch_size)
        
        interact = self.interact_status.iloc[users]
        pos_items = interact[self.col_item + "_interacted"].apply(
            lambda x: random.choice(list(x))
        )
        neg_items = interact[self.col_item + "_interacted"].apply(
            lambda x: sample_neg(x)
        )

        return np.array(users), np.array(pos_items), np.array(neg_items)
    
    @tf.function 
    def train_step(self, users, pos_items, neg_items):
        
        with tf.GradientTape() as tape:
            ua_embeddings, ia_embeddings = self._create_lightgcn_embed()
            
            u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, users)
            pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, pos_items)
            neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, neg_items)

            pos_scores = tf.reduce_sum(tf.multiply(u_g_embeddings, pos_i_g_embeddings), axis=1)
            neg_scores = tf.reduce_sum(tf.multiply(u_g_embeddings, neg_i_g_embeddings), axis=1)

            u_g_embeddings_pre = tf.nn.embedding_lookup(self._weights["user_embedding"], users)
            pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self._weights["item_embedding"], pos_items)
            neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self._weights["item_embedding"], neg_items)

            regularizer = (
                tf.nn.l2_loss(u_g_embeddings_pre)
                + tf.nn.l2_loss(pos_i_g_embeddings_pre)
                + tf.nn.l2_loss(neg_i_g_embeddings_pre)
            )
            regularizer = regularizer / self.batch_size 

            mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
            emb_loss = self.decay * regularizer 
            loss = mf_loss + emb_loss
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, mf_loss, emb_loss 
    
    def fit(self):
        """Fit the model on `self.tr_df`. If `self.eval_epoch` is not -1, evaluate the model on `self.te_df`
        every `self.eval_epoch` epochs to observe the training status.
        """
        for epoch in range(1, self.epochs + 1):
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            n_batch = self.tr_df.shape[0] // self.batch_size + 1

            for _ in range(n_batch):
                users, pos_items, neg_items = self.train_loader(self.batch_size)
                batch_loss, batch_mf_loss, batch_emb_loss = self.train_step(users, pos_items, neg_items)
                loss += batch_loss / n_batch 
                mf_loss += batch_mf_loss / n_batch 
                emb_loss += batch_emb_loss / n_batch 

            # print(f"Epoch {epoch}: Loss={loss:.5f} = (MF_Loss){mf_loss:.5f} + (Emb_Loss){emb_loss:.5f}")

    
    def score(self, user_ids, remove_seen=True):
        """Score all items for test users.

        Args:
            user_ids: numpy.ndarray, users to test
            remove_seen: bool, flag to remove items seen in training from recommendation
        
        Returns:
            numpy.ndarray: value of interest of all items for the users
        """
        ua_embeddings, ia_embeddings = self._create_lightgcn_embed()
        
        u_batch_size = self.batch_size
        n_u_batches = len(user_ids) // u_batch_size + 1
        test_scores = []
        for u_batch_id in range(n_u_batches):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = user_ids[start:end]
            item_batch = range(self.n_items)
            u_emb_batch = tf.nn.embedding_lookup(ua_embeddings, user_batch)
            i_emb_batch = tf.nn.embedding_lookup(ia_embeddings, item_batch)
            rate_batch = tf.matmul(u_emb_batch, i_emb_batch, transpose_b=True)
            test_scores.append(np.array(rate_batch))
        test_scores = np.concatenate(test_scores, axis=0)

        if remove_seen:
            test_scores += self.R.tocsr()[user_ids, :] * -np.inf 
        return test_scores

    def recommend_k_items(self, te_df, top_k=10, sort_top_k=True, remove_seen=True):
        """Recommend top K items for all users in the test set.
        
        Args:
            te_df: pandas.DataFrame, test set
            top_k: int, number of top items to recommend
            sort_top_k: bool, flag to sort top k results
            remove_seen: bool, flag to remove items seen in training from recommendation
        
        Returns:
            pandas.DataFrame, top k recommendation items for each user
        """
        user_ids = np.array(te_df[self.col_user].unique())
        
        test_scores = self.score(user_ids, remove_seen=remove_seen)
        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(
                    # te_df[self.col_user].drop_duplicates().values, top_items.shape[1]
                    te_df[self.col_user].unique(), top_items.shape[1]
                ),
                self.col_item: top_items.flatten(),
                self.col_prediction: top_scores.flatten(),
            }
        )

        return df.replace(-np.inf, np.nan).dropna()
