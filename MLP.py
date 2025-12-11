import random

import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.sparse as sp

from utils import get_top_k_scored_items

class NCF(tf.keras.Model):

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
        
        super(NCF, self).__init__()
        
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed

        self.tr_df = tr_df 
        self.te_df = te_df
        self.n_users = n_users 
        self.n_items = n_items 

        avail_model_types = ["gmf", "mlp", "neumf"]
        if hparams['model_type'] not in avail_model_types:
            raise ValueError(f"`model_type` can only be one of: {avail_model_types}.")
        else:
            self.model_type = hparams['model_type']

        self.epochs = hparams['epochs']
        self.learning_rate= hparams['learning_rate']
        self.batch_size = hparams['batch_size']
        self.n_factors = hparams['n_factors']
        self.layer_sizes = hparams['layer_sizes']
        self.top_k = hparams['top_k']
        self.verbose = hparams['verbose']

        self.col_user = 'user_iid'
        self.col_item = 'item_iid'
        self.col_rating = 'rating'
        self.col_prediction = 'rating_estimate'
        
        # general matrix factorization
        self.user_embedding_gmf = tf.keras.layers.Embedding(self.n_users, self.n_factors)
        self.item_embedding_gmf = tf.keras.layers.Embedding(self.n_items, self.n_factors)

        # multi-layer perceptron
        mlp_dim = self.layer_sizes[0] // 2
        self.user_embedding_mlp = tf.keras.layers.Embedding(self.n_users, mlp_dim)
        self.item_embedding_mlp = tf.keras.layers.Embedding(self.n_items, mlp_dim)

        self.mlp_layers = tf.keras.Sequential()
        for size in self.layer_sizes[1:]:
            self.mlp_layers.add(tf.keras.layers.Dense(size, activation="relu"))

        if self.model_type == "gmf":
            self.predict_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        elif self.model_type == "mlp":
            self.predict_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        else:
            # neumf
            self.predict_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        
        # Initialize interaction history data for `.score()` method
        self._init_train_data()
        
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
        
        # indices = range(self.n_users)
        indices = range(self.interact_status.shape[0])

        # if self.n_users < batch_size:
        if self.interact_status.shape[0] < batch_size:
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
    
    def call(self, inputs):
        user_input, item_input = inputs 
        
        # general matrix factorization
        gmf_users = self.user_embedding_gmf(user_input)
        gmf_items = self.item_embedding_gmf(item_input)
        gmf_vector = tf.multiply(gmf_users, gmf_items)

        # multi-layer perceptron
        mlp_users = self.user_embedding_mlp(user_input)
        mlp_items = self.item_embedding_mlp(item_input)
        mlp_vector = tf.concat([mlp_users, mlp_items], axis=-1)
        mlp_vector = self.mlp_layers(mlp_vector)

        if self.model_type == "gmf":
            output = self.predict_layer(gmf_vector)
        elif self.model_type == "mlp":
            output = self.predict_layer(mlp_vector)
        else:
            # neumf
            neumf_vector = tf.concat([gmf_vector, mlp_vector], axis=-1)
            output = self.predict_layer(neumf_vector)
        return output 
    
    
    def fit(self):
        """
        Train the NCF model using positive and negative sampling.
        
        For each batch, samples users, positive items (items users interacted with),
        and negative items (items users didn't interact with), then trains using BCE loss.
        """
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        # Calculate number of batches per epoch based on training data size
        n_batch = int(np.ceil(len(self.tr_df) / self.batch_size))

        for epoch in range(1, self.epochs+1):
            total_loss = 0.0
            num_batches = 0
            
            for _ in range(n_batch):
                # Sample users, positive items, and negative items
                users, pos_items, neg_items = self.train_loader(self.batch_size)
                
                # Combine positive and negative samples
                # Each user appears twice: once with positive item, once with negative item
                user_batch = np.concatenate([users, users])
                item_batch = np.concatenate([pos_items, neg_items])
                # Labels: 1 for positive samples, 0 for negative samples
                label_batch = np.concatenate([np.ones(len(users)), np.zeros(len(users))])
                
                # Convert to tensors
                user_batch = tf.convert_to_tensor(user_batch, dtype=tf.int32)
                item_batch = tf.convert_to_tensor(item_batch, dtype=tf.int32)
                label_batch = tf.convert_to_tensor(label_batch, dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    preds = self((user_batch, item_batch), training=True)
                    preds = tf.squeeze(preds)  # Remove extra dimension for BCE loss
                    loss = loss_fn(label_batch, preds)
                
                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

                total_loss += loss.numpy()
                num_batches += 1
                
            if self.verbose and epoch % self.verbose == 0:
                print(f"Epoch {epoch}: Loss = {(total_loss / num_batches):.4f}")
    
    def predict(self, user_input, item_input):
        user_input = tf.convert_to_tensor(user_input, dtype=tf.int32)
        item_input = tf.convert_to_tensor(item_input, dtype=tf.int32)

        if len(user_input.shape) == 0:
            # only one user
            user_input = tf.expand_dims(user_input, axis=0)
        if len(item_input.shape) == 0:
            # only one item
            item_input = tf.expand_dims(item_input, axis=0)
        
        preds = self((user_input, item_input), training=False)
        return preds.numpy().flatten() if preds.shape[0] > 1 else float(preds.numpy()[0][0])

    def score(self, user_ids, remove_seen=True):
        """Score all items for test users.

        Args:
            user_ids: numpy.ndarray, users to test
            remove_seen: bool, flag to remove items seen in training from recommendation
        
        Returns:
            numpy.ndarray: value of interest of all items for the users
        """
        u_batch_size = self.batch_size
        n_u_batches = len(user_ids) // u_batch_size + 1
        test_scores = []
        
        for u_batch_id in range(n_u_batches):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = user_ids[start:end]
            
            # Create all item indices for this batch
            item_batch = np.arange(self.n_items)
            
            # Repeat user_batch for each item to create all user-item pairs
            user_repeated = np.repeat(user_batch, self.n_items)
            item_repeated = np.tile(item_batch, len(user_batch))
            
            # Get predictions for all user-item pairs in this batch
            batch_scores = self.predict(user_repeated, item_repeated)
            
            # Reshape to (n_users_in_batch, n_items)
            batch_scores = batch_scores.reshape(len(user_batch), self.n_items)
            test_scores.append(batch_scores)
        
        test_scores = np.concatenate(test_scores, axis=0)

        if remove_seen:
            test_scores += self.R.tocsr()[user_ids, :] * -np.inf 
        
        return np.asarray(test_scores)

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
                    te_df[self.col_user].unique(), top_items.shape[1]
                ),
                self.col_item: top_items.flatten(),
                self.col_prediction: top_scores.flatten(),
            }
        )

        return df.replace(-np.inf, np.nan).dropna()