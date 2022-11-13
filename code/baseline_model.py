import torch
import torch.nn as nn


class IMDBBaseline(nn.Module):
    def __init__(self, embedding_matrix):
        super(IMDBBaseline, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=embedding_matrix.shape[0], embedding_dim=embedding_matrix.shape[1])
        #self.embedding.requires_grad_(False)

        num_embed_dims = embedding_matrix.shape[1]

        self.n_filters = 50

        self.convs = nn.ParameterList([
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=2, padding='same', stride=1),
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=3, padding='same', stride=1),
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=5, padding='same', stride=1)
        ])

        self.relu = nn.ReLU()
        self.cnn_dropout = nn.Dropout(0.5)
        self.layer_out = nn.Linear(150, 1)
        self.tanh = nn.Tanh()

    def forward(self, token_ids, token_mask):
        x = self.embedding(token_ids)  # B, S, D
        x = x * token_mask.unsqueeze(2)
        # print('token_mask', token_mask.shape) # B, S
        x = x.permute(0, 2, 1)  # B,D,S
        conv_outs = [conv(x) for conv in self.convs]
        E = torch.concat(conv_outs, 1)  # B, K, S
        E = E.permute(0, 2, 1)  # B, S, K
        E = self.relu(E)
        E = self.cnn_dropout(E)
        E = E * token_mask.unsqueeze(2)  # B,S,K * B,S,1
        # x = torch.mean(E, 1)  # B, K  # Mean pooling
        x = torch.max(E, 1).values   # Max pooling
        x = self.layer_out(x)  # B, 1
        y_pred = self.tanh(x)
        return y_pred
