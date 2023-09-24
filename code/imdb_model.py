import torch
import torch.nn as nn
from span_predict import SpanPredict


class IMDBModel(nn.Module):
    def __init__(self, embedding_matrix, num_spans: int = 3, freeze_word_embeddings : bool = True):
        super(IMDBModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_word_embeddings)
        num_embed_dims = embedding_matrix.shape[1]

        self.n_filters = 50
        self.num_spans = num_spans
        self.convs = nn.ParameterList([
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=2, padding='same', stride=1),
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=3, padding='same', stride=1),
            nn.Conv1d(in_channels=num_embed_dims, out_channels=self.n_filters, kernel_size=5, padding='same', stride=1)
        ])

        self.K = len(self.convs) * self.n_filters
        if self.num_spans > 0:
            self.span_predict = SpanPredict(num_spans, self.K)

        self.relu = nn.ReLU()
        self.cnn_dropout = nn.Dropout(0.5)
        self.layer_out = nn.Linear(self.K, 1)
        self.tanh = nn.Tanh()

    def forward(self, token_ids, token_mask):
        x = self.embedding(token_ids)  # B, S, D
        x = x * token_mask.unsqueeze(2)
        x = x.permute(0, 2, 1)  # B,D,S
        conv_outs = [conv(x) for conv in self.convs]
        E = torch.concat(conv_outs, 1)  # B, K, S
        E = E.permute(0, 2, 1)  # B, S, K
        E = self.relu(E)
        E = self.cnn_dropout(E)
        E = E * token_mask.unsqueeze(2)  # B,S,K * B,S,1 = B,S,K

        if self.num_spans > 0:
            # span predict model
            r = self.span_predict(E, token_mask)

            M_parts = []
            for ind in range(self.num_spans):
                m = E * r[:, :, ind].unsqueeze(2)  # B,S,K
                # m = torch.mean(m, 1).unsqueeze(2)  # B, K, 1  Mean pooling
                m = torch.max(m, 1).values.unsqueeze(2)  # Max pooling
                m = m.permute(0, 2, 1)  # B,1,K
                M_parts.append(m)
            M = torch.concat(M_parts, 1)  # B,J,K
            z = self.layer_out(M)  # B, J, 1
            z = self.tanh(z)
            y_pred = torch.sum(z, 1)  # B, 1
        else:
            # baseline model
            M = torch.max(E, 1).values  # Max pooling
            x = self.layer_out(M)  # B, 1
            y_pred = self.tanh(x)
            r = None

        return y_pred, r
