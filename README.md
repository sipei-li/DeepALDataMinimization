# Fairness of Active Learning with Deep Recommender Systems

## Rec Sys
* LightGCN
    * A GNN-based deep rec sys, implemented with tensorflow in `LightGCN.py`
* NeuMF
    * A neural network-based generalization of matrix factorization, implemented with tensorflow in `MLP.py`

## AL
* Personalized and non-personalized strategies, implemented in `ActiveLearner.py`
* Parameters that are used for experiments
    * `oversample`: 
        * oversample female data to match the amount of male data
        * Not so useful because LightGCN uses a user-item matrix to train the model
    * `ratio`: 
        * downsample male data or oversample female data to make certain male data to female data ratio
        *  Used in Experiment 2
    * `low_rate_extend`: 
        * whether to extend the query windows for low response rate users
        * Used in Experiment 5
    * `low_rate_extend_percentage`: 
        * percentage of bottom users to be considered as low response rate users
        * Used in Experiment 5
    * `low_rate_extend_gamma`
        * Maximum possible window size after extension
        * Used in Experiment 5

## Experiements

* Experiment 1 (`*_exp1.ipynb`)
    * Show the bias of personalized and non-personalized AL strategies with LightGCN
* Experiment 2 (`*_exp2.ipynb`)
    * Equal rating count and equal response rate experiments for personalized AL with LightGCN
* Experiment 3 (`*_exp3.ipynb`)
    * Separate female and male data experiment (not important)
* Experiment 4 (`*_exp4.ipynb`)
    * Batch Fairness Extend experiments (discontinued)
* Experiment 5 (`*_exp5.ipynb`)
    * Extend query windows for low response rate users (currently doing)
