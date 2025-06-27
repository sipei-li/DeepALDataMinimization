import random

import numpy as np
import pandas as pd
import tensorflow as tf 

class NCF(tf.keras.Model):

    def __init__(self, hparams, tr_df, te_df, n_users, n_items, seed=None):

        super(NCF, self).__init__()

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed 

        avail_model_types = ["gmf", "mlp", "neumf"]
        if hparams['model_type'] not in avail_model_types:
            raise ValueError(f"`model_type` can only be one of: {avail_model_types}.")
        else:
            self.model_type = hparams['model_type']
        
        self.n_factors = hparams['n_factors']
        self.layer_sizes = hparams['layer_sizes']
        self.n_epochs = hparams['n_epochs']
        self.batch_size = hparams['batch_size']
        self.learning_rate = hparams['learning_rate']

        self.tr_df = tr_df 
        self.te_df = te_df 
        self.n_users = n_users 
        self.n_items = n_items 

        # Inputs
        user_input = tf.keras.layers.Input(shape=(1,), name="user_input")
        item_input = tf.keras.layers.Input(shape=(1,), name="item_input")
