import random

import numpy as np
import tensorflow as tf

class NCF(tf.keras.Model):

    def __init__(self, hparams, n_users, n_items, seed=None):
        super().__init__()
        
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
        self.verbose = hparams['verbose']
        self.batch_size = hparams['batch_size']
        self.learning_rate= hparams['learning_rate']

        # general matrix factorization
        self.user_embedding_gmf = tf.keras.layers.Embedding(n_users, self.n_factors)
        self.item_embedding_gmf = tf.keras.layers.Embedding(n_items, self.n_factors)

        # multi-layer perceptron
        mlp_dim = self.layer_sizes[0] // 2
        self.user_embedding_mlp = tf.keras.layers.Embedding(n_users, mlp_dim)
        self.item_embedding_mlp = tf.keras.layers.Embedding(n_items, mlp_dim)

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
    
    def fit(self, train_data, epochs=10, learning_rate=0.001, verbose=1):
        """
        Train the NCF model.

        Args:
            train_data (tf.data.Dataset or generator): ((user_batch, item_batch), label_batch)
            epochs (int): number of training epochs.
            learning_rate (float): learning rate for optimizer.
            verbose (int): how often to print loss.
        """
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        for epoch in range(1, epochs+1):
            total_loss = 0.0
            num_batches = 0
            for (user_batch, item_batch), label_batch in train_data:
                with tf.GradientTape() as tape:
                    preds = self((user_batch, item_batch), training=True)
                    loss = loss_fn(label_batch, preds)
                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

                total_loss += loss.numpy()
                num_batches += 1
            if verbose and epoch % verbose == 0:
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