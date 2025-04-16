import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, user_n, item_n, factor_num, layers):
        super(MLP, self).__init__()
        self.user_n = user_n
        self.item_n = item_n
        self.factor_num = factor_num
        self.layers = layers

        self.user_embeddings = nn.Embedding(num_embeddings=self.user_n, embedding_dim=self.factor_num)
        self.item_embeddings = nn.Embedding(num_embeddings=self.item_n, embedding_dim=self.factor_num)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        
        self.affine_output = nn.Linear(in_features=self.layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, u_ids, i_ids):
        user_embeds = self.user_embeddings(u_ids)
        item_embeds = self.item_embeddings(i_ids)
        # concatenate user and item embeddings together
        input = torch.cat([user_embeds, item_embeds], dim=-1)
        for idx in range(len(self.fc_layers)):
            input = self.fc_layers[idx](input)
            input = nn.ReLU()(input)
        
        logits = self.affine_output(input)
        rating = self.logistic(logits)
        return rating
    
    def init_weight(slef):
        pass